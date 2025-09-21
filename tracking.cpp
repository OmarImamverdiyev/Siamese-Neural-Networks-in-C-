// tracking.cpp
// Minimal SiamTPN-style ONNX tracker with OTB precision benchmark (C++17 + OpenCV + ONNX Runtime)
//
// Build:
//     mkdir build
//     cd build
//     cmake .. 
//     make -j
//   
// Demo:
//   ./tracking --video ./test.mp4 --z ./backbone_fpn_z.onnx --x ./backbone_fpn_head_x.onnx --show --save ./tracked.avi
//
// Benchmark:
//   ./tracking --otb-root ../OTB --z ../backbone_fpn_z.onnx --x ../backbone_fpn_head_x.onnx
//
// Notes:
//   * CPU by default. If your ORT build has CUDA EP, you can enable it where indicated.

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <filesystem>
#include <regex>
#include <iomanip>
#include <sstream>

// helpers 
static inline void die_if(bool cond, const std::string& msg) {
    if (cond) throw std::runtime_error(msg);
}
static inline std::string shape_str(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) { out += std::to_string(s[i]); if (i+1<s.size()) out+='x'; }
    out += "]";
    return out;
}
static inline bool valid_num(float v){ return std::isfinite(v); }
static inline bool valid_box(const cv::Rect2f& r){
    return valid_num(r.x) && valid_num(r.y) && valid_num(r.width) && valid_num(r.height);
}

// config 
struct Cfg {
    float TEMPLATE_FACTOR = 1.5f;
    int   TEMPLATE_SIZE   = 320;   
    float SEARCH_FACTOR   = 5.0f;
    int   SEARCH_SIZE     = 320;   
    float HANNING_FACTOR  = 0.01f;
    float ANCHOR_BIAS     = 0.5f;
    bool  BOX_NORMALIZED  = true;   
    float SMOOTH_ALPHA    = 0.6f;   
    float ANCHOR_FACTOR   = 1.0f;
};

//preprocessing 
struct PreprocessorX_ONNX {
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> process(const cv::Mat& rgb, int& H, int& W) const {
        CV_Assert(rgb.type() == CV_8UC3);
        H = rgb.rows; W = rgb.cols;
        std::vector<float> out(1 * 3 * H * W);
        const int strideC = H * W;
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = rgb.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                float r = row[3*x+0] / 255.f;
                float g = row[3*x+1] / 255.f;
                float b = row[3*x+2] / 255.f;
                out[0*strideC + y*W + x] = (r - mean[0]) / stdv[0];
                out[1*strideC + y*W + x] = (g - mean[1]) / stdv[1];
                out[2*strideC + y*W + x] = (b - mean[2]) / stdv[2];
            }
        }
        return out;
    }
};

// crop and resize around target 
static std::pair<cv::Mat, float> sample_target_fast(const cv::Mat& img_rgb,
                                                    const cv::Rect2f& bb_xywh,
                                                    float search_area_factor,
                                                    int output_sz) {
    float w = std::max(1.0f, bb_xywh.width);
    float h = std::max(1.0f, bb_xywh.height);
    int crop_sz = std::max(1, (int)std::round(std::sqrt(w*h) * search_area_factor));

    float cx = bb_xywh.x + 0.5f * bb_xywh.width;
    float cy = bb_xywh.y + 0.5f * bb_xywh.height;
    int x1 = (int)std::round(cx - 0.5f * crop_sz);
    int y1 = (int)std::round(cy - 0.5f * crop_sz);
    int x2 = x1 + crop_sz;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int y1_pad = std::max(0, -y1);
    int x2_pad = std::max(0, x2 - img_rgb.cols + 1);
    int y2_pad = std::max(0, y2 - img_rgb.rows + 1);

    cv::Rect roi(x1 + x1_pad, y1 + y1_pad,
                 crop_sz - x1_pad - x2_pad,
                 crop_sz - y1_pad - y2_pad);
    roi &= cv::Rect(0, 0, img_rgb.cols, img_rgb.rows);

    cv::Mat im_crop = img_rgb(roi).clone();
    cv::Mat padded;
    cv::copyMakeBorder(im_crop, padded, y1_pad, y2_pad, x1_pad, x2_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    float resize_factor = (float)output_sz / (float)crop_sz;
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(output_sz, output_sz), 0, 0, cv::INTER_LINEAR);
    return {resized, resize_factor};
}

// clip box 
static cv::Rect2f clip_box(const cv::Rect2f& box, int H, int W, float margin=0.0f) {
    float x1 = box.x, y1 = box.y, x2 = x1 + box.width, y2 = y1 + box.height;
    x1 = std::min(std::max(0.f, x1), (float)W - margin);
    y1 = std::min(std::max(0.f, y1), (float)H - margin);
    x2 = std::min(std::max(margin, x2), (float)W);
    y2 = std::min(std::max(margin, y2), (float)H);
    float w = std::max(margin, x2 - x1);
    float h = std::max(margin, y2 - y1);
    return {x1, y1, w, h};
}

// Hanning window 
static std::vector<float> hanning_window(int num) {
    if (num <= 0) return {};
    const float PI = 3.14159265358979323846f;
    std::vector<float> w(num);
    if (num == 1) w[0] = 1.0f;
    else for (int i = 0; i < num; ++i) w[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / (float)(num - 1));
    std::vector<float> hw; hw.reserve(num * num);
    for (int y = 0; y < num; ++y) for (int x = 0; x < num; ++x) hw.push_back(w[y] * w[x]);
    return hw;
}

//  anchors 
static std::vector<cv::Point2f> make_anchors(int grid, const Cfg& cfg) {
    std::vector<cv::Point2f> g; g.reserve(grid * grid);
    for (int yy = 0; yy < grid; ++yy) {
        for (int xx = 0; xx < grid; ++xx) {
            if (cfg.BOX_NORMALIZED) {
                float gx = (cfg.ANCHOR_FACTOR * (xx) + cfg.ANCHOR_BIAS) / (float)grid;
                float gy = (cfg.ANCHOR_FACTOR * (yy) + cfg.ANCHOR_BIAS) / (float)grid;
                g.emplace_back(gx, gy);
            } else {
                float stride = (float)cfg.SEARCH_SIZE / (float)grid;
                float gx = (xx + cfg.ANCHOR_BIAS) * stride;
                float gy = (yy + cfg.ANCHOR_BIAS) * stride;
                g.emplace_back(gx, gy);
            }
        }
    }
    return g;
}

//  ONNX helpers
struct IoNames {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

static IoNames get_io_names(Ort::Session& sess, Ort::AllocatorWithDefaultOptions& alloc) {
    IoNames n;
    size_t ni = sess.GetInputCount();
    size_t no = sess.GetOutputCount();
    for (size_t i = 0; i < ni; ++i) {
        auto s = sess.GetInputNameAllocated(i, alloc);
        n.inputs.emplace_back(s.get());
    }
    for (size_t i = 0; i < no; ++i) {
        auto s = sess.GetOutputNameAllocated(i, alloc);
        n.outputs.emplace_back(s.get());
    }
    return n;
}

static std::vector<int64_t> get_io_shape(Ort::Session& sess, size_t index, bool input) {
    Ort::TypeInfo ti = input ? sess.GetInputTypeInfo(index) : sess.GetOutputTypeInfo(index);
    auto tinfo = ti.GetTensorTypeAndShapeInfo();
    return tinfo.GetShape();
}

static int get_model_input_hw(Ort::Session& sess, int fallback) {
    size_t ni = sess.GetInputCount();
    for (size_t i = 0; i < ni; ++i) {
        auto shp = get_io_shape(sess, i, true);
        if (shp.size() == 4) {
            int64_t H = shp[2], W = shp[3];
            if (H > 0 && W > 0) return (int)H; 
        }
    }
    return fallback;
}

static void squeeze_leading_ones(std::vector<int64_t>& s) {
    while (!s.empty() && s.front() == 1) s.erase(s.begin());
}

// tracker
class SiamTPN_ONNX {
public:
    SiamTPN_ONNX(const std::string& z_path, const std::string& x_path, Cfg cfg)
        : cfg_(cfg),
          env_(ORT_LOGGING_LEVEL_WARNING, "siamtpn"),
          mem_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
    {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        // If your ORT build has CUDA EP, enable:
        // OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0);

        sess_z_ = std::make_unique<Ort::Session>(env_, z_path.c_str(), so);
        sess_x_ = std::make_unique<Ort::Session>(env_, x_path.c_str(), so);
        names_z_ = get_io_names(*sess_z_, alloc_);
        names_x_ = get_io_names(*sess_x_, alloc_);

        die_if(names_z_.inputs.size() != 1, "Template model must have exactly 1 input.");
        die_if(names_x_.inputs.size() != 2, "Search model must have exactly 2 inputs.");
        die_if(names_x_.outputs.size() < 2, "Search model must have at least 2 outputs.");

        int z_hw = get_model_input_hw(*sess_z_, cfg_.TEMPLATE_SIZE);
        int x_hw = get_model_input_hw(*sess_x_, cfg_.SEARCH_SIZE);
        cfg_.TEMPLATE_SIZE = z_hw;
        cfg_.SEARCH_SIZE   = x_hw;
    }

    void initialize(const cv::Mat& image_rgb, const cv::Rect2f& init_xywh) {
        state_ = init_xywh;

        auto tpl = sample_target_fast(image_rgb, state_, cfg_.TEMPLATE_FACTOR, cfg_.TEMPLATE_SIZE);
        cv::Mat z_patch = tpl.first;

        int H, W;
        auto z_blob = pre_.process(z_patch, H, W);
        std::vector<int64_t> z_shape = {1, 3, H, W};
        Ort::Value z_tensor = Ort::Value::CreateTensor<float>(mem_, z_blob.data(), z_blob.size(),
                                                              z_shape.data(), z_shape.size());

        std::array<const char*, 1> in_names = { names_z_.inputs[0].c_str() };
        std::vector<const char*> out_names; out_names.reserve(names_z_.outputs.size());
        for (auto& s : names_z_.outputs) out_names.push_back(s.c_str());
        std::array<Ort::Value, 1> inputs = { std::move(z_tensor) };

        auto outs = sess_z_->Run(Ort::RunOptions{nullptr},
                                 in_names.data(),
                                 inputs.data(), inputs.size(),
                                 out_names.data(), out_names.size());
        die_if(outs.empty(), "z-session returned no outputs.");

        auto& ker_val = outs[0];
        auto k_info = ker_val.GetTensorTypeAndShapeInfo();
        kernel_shape_ = k_info.GetShape();
        size_t numel = k_info.GetElementCount();
        kernel_buf_.resize(numel);
        const float* src = ker_val.GetTensorData<float>();
        std::copy(src, src + numel, kernel_buf_.begin());

        frame_counter_ = 0;
    }

    cv::Rect2f track(const cv::Mat& image_rgb) {
        auto sr = sample_target_fast(image_rgb, state_, cfg_.SEARCH_FACTOR, cfg_.SEARCH_SIZE);
        cv::Mat x_patch = sr.first;
        float resize_factor = sr.second;

        int H, W;
        auto x_blob = pre_.process(x_patch, H, W);
        std::vector<int64_t> x_shape = {1, 3, H, W};
        Ort::Value x_tensor = Ort::Value::CreateTensor<float>(mem_, x_blob.data(), x_blob.size(),
                                                              x_shape.data(), x_shape.size());
        Ort::Value k_tensor = Ort::Value::CreateTensor<float>(mem_, kernel_buf_.data(), kernel_buf_.size(),
                                                              kernel_shape_.data(), kernel_shape_.size());

        std::array<const char*, 2> in_names = {
            names_x_.inputs[0].c_str(),
            names_x_.inputs[1].c_str()
        };
        std::vector<const char*> out_names; out_names.reserve(names_x_.outputs.size());
        for (auto& s : names_x_.outputs) out_names.push_back(s.c_str());
        std::array<Ort::Value, 2> inputs = { std::move(x_tensor), std::move(k_tensor) };

        auto outs = sess_x_->Run(Ort::RunOptions{nullptr},
                                 in_names.data(),
                                 inputs.data(), inputs.size(),
                                 out_names.data(), out_names.size());

        int idx_scores = -1, idx_boxes = -1;
        for (size_t i = 0; i < outs.size(); ++i) {
            auto info = outs[i].GetTensorTypeAndShapeInfo();
            auto shp = info.GetShape();
            squeeze_leading_ones(shp);
            if (shp.size() == 2 && shp[1] == 2) idx_scores = (int)i;
            if (shp.size() == 2 && shp[1] == 4) idx_boxes  = (int)i;
        }
        die_if(idx_scores < 0 || idx_boxes < 0, "Could not locate scores (..,2) and boxes (..,4).");

        // if (frame_counter_ < 5) {
        //     auto s_info = outs[idx_scores].GetTensorTypeAndShapeInfo();
        //     auto b_info = outs[idx_boxes ].GetTensorTypeAndShapeInfo();
        //     std::cerr << "[dbg] scores=" << shape_str(s_info.GetShape())
        //               << "  boxes=" << shape_str(b_info.GetShape())
        //               << "  BOX_NORMALIZED=" << (cfg_.BOX_NORMALIZED ? 1 : 0) << "\n";
        // }

        auto s_info = outs[idx_scores].GetTensorTypeAndShapeInfo();
        auto s_shape = s_info.GetShape(); squeeze_leading_ones(s_shape);
        size_t N = (size_t)s_shape[0];
        const float* scores_ptr = outs[idx_scores].GetTensorData<float>();
        std::vector<float> scores(scores_ptr, scores_ptr + N * 2);
        std::vector<float> cls1(N);
        for (size_t i = 0; i < N; ++i) cls1[i] = scores[2*i + 1];

        auto b_info = outs[idx_boxes].GetTensorTypeAndShapeInfo();
        auto b_shape = b_info.GetShape(); squeeze_leading_ones(b_shape);
        die_if((size_t)b_shape[0] != N || b_shape[1] != 4, "Boxes shape mismatch.");
        const float* boxes_ptr = outs[idx_boxes].GetTensorData<float>();

        if (winN_ != (int)N) {
            int grid = (int)std::round(std::sqrt((double)N));
            window_ = hanning_window(grid);
            if ((int)window_.size() != grid*grid) window_.assign(N, 1.0f);
            anchors_ = make_anchors(grid, cfg_);
            if ((int)anchors_.size() != grid*grid) anchors_.resize(N, cv::Point2f(0,0));
            winN_ = (int)N;
        }

        if (cfg_.HANNING_FACTOR > 0.f && (int)window_.size() == winN_) {
            for (size_t i = 0; i < N; ++i)
                cls1[i] = cls1[i] * (1.0f - cfg_.HANNING_FACTOR) + cfg_.HANNING_FACTOR * window_[i];
        }

        int best = 0; float bestv = -1e30f;
        for (size_t i = 0; i < N; ++i) if (cls1[i] > bestv) { bestv = cls1[i]; best = (int)i; }

        const float* bb = boxes_ptr + best * 4;
        float l = bb[0], t = bb[1], r = bb[2], b = bb[3];

        float gx = anchors_[best].x;
        float gy = anchors_[best].y;

        float x1_crop, y1_crop, x2_crop, y2_crop;
        if (cfg_.BOX_NORMALIZED) {
            x1_crop = (gx - l) * cfg_.SEARCH_SIZE;
            y1_crop = (gy - t) * cfg_.SEARCH_SIZE;
            x2_crop = (gx + r) * cfg_.SEARCH_SIZE;
            y2_crop = (gy + b) * cfg_.SEARCH_SIZE;
        } else {
            x1_crop = gx - l; y1_crop = gy - t; x2_crop = gx + r; y2_crop = gy + b;
        }

        if (x2_crop < x1_crop) std::swap(x1_crop, x2_crop);
        if (y2_crop < y1_crop) std::swap(y1_crop, y2_crop);

        float x1 = x1_crop / resize_factor;
        float y1 = y1_crop / resize_factor;
        float x2 = x2_crop / resize_factor;
        float y2 = y2_crop / resize_factor;

        float cx_prev = state_.x + 0.5f * state_.width;
        float cy_prev = state_.y + 0.5f * state_.height;
        float half_side = 0.5f * (float)cfg_.SEARCH_SIZE / resize_factor;
        float crop_tl_x = cx_prev - half_side;
        float crop_tl_y = cy_prev - half_side;

        x1 += crop_tl_x;  y1 += crop_tl_y;
        x2 += crop_tl_x;  y2 += crop_tl_y;

        cv::Rect2f new_state(x1, y1, std::max(0.f, x2 - x1), std::max(0.f, y2 - y1));

        const float MIN_WH = 4.0f;
        if (new_state.width < MIN_WH)  new_state.width  = MIN_WH;
        if (new_state.height < MIN_WH) new_state.height = MIN_WH;

        if (cfg_.SMOOTH_ALPHA > 0.f && frame_counter_ > 0) {
            float a = cfg_.SMOOTH_ALPHA;
            new_state.x      = a*state_.x      + (1.f-a)*new_state.x;
            new_state.y      = a*state_.y      + (1.f-a)*new_state.y;
            new_state.width  = a*state_.width  + (1.f-a)*new_state.width;
            new_state.height = a*state_.height + (1.f-a)*new_state.height;
        }

        new_state = clip_box(new_state, image_rgb.rows, image_rgb.cols, 10.0f);
        if (!valid_box(new_state) || new_state.width < MIN_WH || new_state.height < MIN_WH) {
            new_state = state_;
        }

        state_ = new_state;
        ++frame_counter_;
        return state_;
    }

    const Cfg& cfg() const { return cfg_; }

private:
    Cfg cfg_;
    PreprocessorX_ONNX pre_;

    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions alloc_;
    Ort::MemoryInfo mem_;
    std::unique_ptr<Ort::Session> sess_z_;
    std::unique_ptr<Ort::Session> sess_x_;
    IoNames names_z_, names_x_;

    std::vector<float> kernel_buf_;
    std::vector<int64_t> kernel_shape_;

    cv::Rect2f state_;
    int frame_counter_ = 0;

    int winN_ = -1;
    std::vector<float> window_;
    std::vector<cv::Point2f> anchors_;
};

// UI helper 
static void draw_help(cv::Mat& img) {
    cv::rectangle(img, {0,0}, {img.cols, 50}, {255,255,255}, cv::FILLED);
    cv::putText(img, "Select ROI (drag). ENTER/SPACE=OK; ESC=cancel",
                {12, 32}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,0,0}, 2, cv::LINE_AA);
}

// Precision utilities 
static float center_distance(const cv::Rect2f& a, const cv::Rect2f& b) {
    float cx1 = a.x + 0.5f * a.width,  cy1 = a.y + 0.5f * a.height;
    float cx2 = b.x + 0.5f * b.width,  cy2 = b.y + 0.5f * b.height;
    float dx = cx1 - cx2, dy = cy1 - cy2;
    return std::sqrt(dx*dx + dy*dy);
}

static bool parse_gt_line(const std::string& line, cv::Rect2f& r) {
    std::string s = line; std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream iss(s);
    float x,y,w,h;
    if (iss >> x >> y >> w >> h) { r = {x,y,w,h}; return true; }
    return false;
}

static std::vector<cv::Rect2f> load_otb_gt_any(const std::string& seq_dir) {
    namespace fs = std::filesystem;
    const char* cands[] = {"groundtruth_rect.txt","groundtruth_rect.1.txt","groundtruth_rect.2.txt"};
    std::string gt_path;
    for (auto c : cands) {
        auto p = (fs::path(seq_dir) / c).string();
        if (fs::exists(p)) { gt_path = p; break; }
    }
    if (gt_path.empty()) throw std::runtime_error("No groundtruth_rect*.txt under: " + seq_dir);

    std::ifstream fin(gt_path);
    if (!fin.is_open()) throw std::runtime_error("Failed to open GT: " + gt_path);
    std::vector<cv::Rect2f> out; std::string line;
    while (std::getline(fin, line)) { if (line.empty()) continue; cv::Rect2f r; if (parse_gt_line(line, r)) out.push_back(r); }
    if (out.empty()) throw std::runtime_error("No GT boxes in: " + gt_path);
    return out;
}

static std::vector<std::string> list_frames(const std::string& img_dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    static const std::regex rx(R"(.*\.(jpg|jpeg|png|bmp|JPG|JPEG|PNG|BMP))");
    if (!fs::exists(img_dir)) throw std::runtime_error("Missing img/ dir: " + img_dir);
    for (auto& de : fs::directory_iterator(img_dir)) {
        if (!de.is_regular_file()) continue;
        auto p = de.path().string();
        if (std::regex_match(p, rx)) files.push_back(p);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("No images under: " + img_dir);
    return files;
}

static std::vector<std::string> list_otb_sequences(const std::string& otb_root) {
    namespace fs = std::filesystem;
    std::vector<std::string> seqs;
    if (!fs::exists(otb_root)) throw std::runtime_error("OTB root not found: " + otb_root);
    for (auto& de : fs::directory_iterator(otb_root)) {
        if (!de.is_directory()) continue;
        const auto seq_dir = de.path().string();
        if (fs::exists(fs::path(seq_dir) / "img") &&
            (fs::exists(fs::path(seq_dir) / "groundtruth_rect.txt") ||
             fs::exists(fs::path(seq_dir) / "groundtruth_rect.1.txt") ||
             fs::exists(fs::path(seq_dir) / "groundtruth_rect.2.txt")))
            seqs.push_back(de.path().filename().string());
    }
    std::sort(seqs.begin(), seqs.end());
    return seqs;
}

static double precision_at(const std::vector<cv::Rect2f>& preds,
                           const std::vector<cv::Rect2f>& gts, float thr_px) {
    const size_t n = std::min(preds.size(), gts.size());
    if (n == 0) return 0.0;
    size_t ok = 0;
    for (size_t i=0;i<n;++i) if (center_distance(preds[i], gts[i]) <= thr_px) ++ok;
    return 100.0 * double(ok) / double(n);
}

static int run_otb_precision_benchmark(const std::string& otb_root,
                                       const std::string& z_path,
                                       const std::string& x_path,
                                       float primary_thr_px) {
    Cfg cfg; 
    SiamTPN_ONNX tracker(z_path, x_path, cfg);

    auto seqs = list_otb_sequences(otb_root);
    if (seqs.empty()) { std::cerr << "No OTB sequences under " << otb_root << "\n"; return 1; }

    std::vector<int> thrs = {5, 10, 20, 30, 50};
    std::cout << "=== OTB Precision Benchmark ===\n";
    std::cout << "Root: " << otb_root << " | Primary thr(px)=" << primary_thr_px << "\n\n";
    std::cout << std::left << std::setw(20) << "Sequence";
    for (int t : thrs) std::cout << std::right << std::setw(10) << ("@" + std::to_string(t) + "px");
    std::cout << "\n" << std::string(20 + 10*thrs.size(), '-') << "\n";

    double sums[5] = {0,0,0,0,0};
    int done = 0;

    for (const auto& name : seqs) {
        try {
            const std::string seq_dir = (std::filesystem::path(otb_root) / name).string();
            const std::string img_dir = (std::filesystem::path(seq_dir) / "img").string();

            std::vector<cv::Rect2f> gt = load_otb_gt_any(seq_dir);
            auto frames = list_frames(img_dir);

            const size_t T = std::min(frames.size(), gt.size());
            if (T < 2) { std::cerr << "WARN: too few frames/gt in " << name << "\n"; continue; }

            cv::Mat f0 = cv::imread(frames[0]);
            if (f0.empty()) { std::cerr << "WARN: cannot read " << frames[0] << " (skip " << name << ")\n"; continue; }
            cv::Mat f0_rgb; cv::cvtColor(f0, f0_rgb, cv::COLOR_BGR2RGB);
            tracker.initialize(f0_rgb, gt[0]);

            std::vector<cv::Rect2f> preds; preds.reserve(T);
            preds.push_back(gt[0]); 
            for (size_t i = 1; i < T; ++i) {
                cv::Mat fr = cv::imread(frames[i]); if (fr.empty()) break;
                cv::Mat fr_rgb; cv::cvtColor(fr, fr_rgb, cv::COLOR_BGR2RGB);
                preds.push_back(tracker.track(fr_rgb));
            }

            double row[5];
            for (int k=0;k<(int)thrs.size();++k) {
                row[k] = precision_at(preds, gt, float(thrs[k]));
                sums[k] += row[k];
            }
            ++done;

            std::cout << std::left << std::setw(20) << name;
            for (int k=0;k<(int)thrs.size();++k)
                std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << row[k];
            std::cout << "\n";
        } catch (const std::exception& e) {
            std::cerr << "WARN [" << name << "]: " << e.what() << " (skipped)\n";
        }
    }

    if (done > 0) {
        std::cout << std::string(20 + 10*thrs.size(), '-') << "\n";
        std::cout << std::left << std::setw(20) << "Average";
        for (int k=0;k<(int)thrs.size();++k)
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << (sums[k]/done);
        std::cout << "\n\nPrimary Precision@" << (int)primary_thr_px << "px: "
                  << std::fixed << std::setprecision(2) << (sums[2] / done) << "%\n"; // 20px index
    } else {
        std::cout << "No sequences evaluated.\n";
    }
    return 0;
}


int main(int argc, char** argv) {
    try {
        // Common args
        std::string z_path = "./backbone_fpn_z.onnx";
        std::string x_path = "./backbone_fpn_head_x.onnx";

        // Demo (video) args
        std::string video = "./test.mp4";
        std::string save_path;
        bool show = true;
        Cfg cfg; // baked defaults already set

        // Benchmark (dataset) args
        std::string otb_root;   
        float primary_thr_px = 20.0f;

        
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const char* f){ die_if(i+1>=argc, std::string("Missing value for ")+f); return std::string(argv[++i]); };
            if (a == "--z") z_path = next("--z");
            else if (a == "--x") x_path = next("--x");
            else if (a == "--otb-root")  otb_root = next("--otb-root");
            else if (a == "--thr")       primary_thr_px = std::stof(next("--thr"));
            else if (a == "--video")     video = next("--video");
            else if (a == "--save")      save_path = next("--save");
            else if (a == "--show")      show = true;
            else if (a == "--no-show")   show = false;
            // You can still override baked defaults if desired:
            else if (a == "--box-normalized") cfg.BOX_NORMALIZED = (std::stoi(next("--box-normalized")) != 0);
            else if (a == "--hanning")   cfg.HANNING_FACTOR = std::stof(next("--hanning"));
            else if (a == "--smooth")    cfg.SMOOTH_ALPHA   = std::stof(next("--smooth"));
            else if (a == "--template-factor") cfg.TEMPLATE_FACTOR = std::stof(next("--template-factor"));
            else if (a == "--force-template")  cfg.TEMPLATE_SIZE   = std::stoi(next("--force-template"));
            else if (a == "--force-search")    cfg.SEARCH_SIZE     = std::stoi(next("--force-search"));
            else {
                std::cerr << "Unknown arg: " << a << "\n";
                return 2;
            }
        }

        // BENCHMARK MODE 
        if (!otb_root.empty()) {
            die_if(!std::ifstream(z_path).good(), "File not found: " + z_path);
            die_if(!std::ifstream(x_path).good(), "File not found: " + x_path);
            return run_otb_precision_benchmark(otb_root, z_path, x_path, primary_thr_px);
        }

        //  DEMO MODE 
        die_if(!std::ifstream(video).good(), "Video not found: " + video);
        die_if(!std::ifstream(z_path).good(), "File not found: " + z_path);
        die_if(!std::ifstream(x_path).good(), "File not found: " + x_path);

        cv::VideoCapture cap(video);
        die_if(!cap.isOpened(), "Failed to open video.");
        cv::Mat frame_bgr; die_if(!cap.read(frame_bgr) || frame_bgr.empty(), "Failed to read first frame.");

        cv::Mat first = frame_bgr.clone();
        draw_help(first);
        cv::Rect2d roi = cv::selectROI("Select ROI", first, /*showCrosshair*/true, /*fromCenter*/false);
        cv::destroyWindow("Select ROI");
        die_if(roi.width <= 0 || roi.height <= 0, "ROI selection cancelled or invalid.");

        SiamTPN_ONNX tracker(z_path, x_path, cfg);

        cv::Mat frame_rgb; cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
        tracker.initialize(frame_rgb, cv::Rect2f((float)roi.x,(float)roi.y,(float)roi.width,(float)roi.height));

        std::unique_ptr<cv::VideoWriter> writer;
        if (!save_path.empty()) {
            int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
            double fps = cap.get(cv::CAP_PROP_FPS); if (fps <= 1e-3) fps = 25.0;
            writer = std::make_unique<cv::VideoWriter>(save_path, fourcc, fps,
                        cv::Size(frame_bgr.cols, frame_bgr.rows));
            die_if(!writer->isOpened(), "Failed to open VideoWriter for: " + save_path);
        }

        while (true) {
            if (!cap.read(frame_bgr) || frame_bgr.empty()) break;
            cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

            int64 t0 = cv::getTickCount();
            cv::Rect2f bb = tracker.track(frame_rgb);
            double fps = cv::getTickFrequency() / (cv::getTickCount() - t0);

            cv::Mat out = frame_bgr.clone();
            cv::Point c((int)(bb.x + bb.width*0.5f), (int)(bb.y + bb.height*0.5f));
            cv::drawMarker(out, c, {0,255,255}, cv::MARKER_CROSS, 12, 2);
            cv::rectangle(out, bb, {0,255,0}, 3);
            char txt[64]; std::snprintf(txt, sizeof(txt), "FPS: %.1f", fps);
            cv::putText(out, txt, {12,28}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {10,10,10}, 2, cv::LINE_AA);

            if (writer) writer->write(out);
            if (show) {
                cv::imshow("Tracking", out);
                int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') break;
            }
        }
        if (show) cv::destroyAllWindows();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}

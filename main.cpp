#include <torch/script.h> // LibTorch 頭文件
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "stixel_world.h"
// 全域變數，用於鼠標回調函數訪問
cv::Mat disp;

// 鼠標事件回調函數
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // 獲取視差值
        float disparity = disp.at<float>(y, x);
        std::cout << "視差值於 (" << x << ", " << y << "): " << disparity << std::endl;
    }
}

int main() {
    // 載入 TorchScript 模型
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("stereo_model.pt");
        module.to(torch::kCUDA); // 將模型移至 CUDA
    }
    catch (const c10::Error& e) {
        std::cerr << "載入模型時發生錯誤。\n";
        return -1;
    }

    // 讀取輸入影像
    cv::Mat imgL = cv::imread("/home/ab123/Desktop/thesis_9_5/L_1.png", cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread("/home/ab123/Desktop/thesis_9_5/R_1.png", cv::IMREAD_COLOR);

    if (imgL.empty() || imgR.empty()) {
        std::cerr << "無法讀取其中一張圖片。\n";
        return -1;
    }

    // 將影像從 BGR 轉換為 RGB
    cv::cvtColor(imgL, imgL, cv::COLOR_BGR2RGB);
    cv::cvtColor(imgR, imgR, cv::COLOR_BGR2RGB);

    // 將影像轉換為 float 並縮放至 [0,1]
    imgL.convertTo(imgL, CV_32FC3, 1.0 / 255.0);
    imgR.convertTo(imgR, CV_32FC3, 1.0 / 255.0);

    // 使用 ImageNet 的平均值和標準差進行標準化
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);

    // 減去平均值並除以標準差
    cv::Mat imgL_norm, imgR_norm;
    cv::subtract(imgL, mean, imgL_norm);
    cv::divide(imgL_norm, std, imgL_norm);

    cv::subtract(imgR, mean, imgR_norm);
    cv::divide(imgR_norm, std, imgR_norm);

    // 將影像轉換為 torch 張量
    torch::Tensor tensorL = torch::from_blob(imgL_norm.data, {1, imgL_norm.rows, imgL_norm.cols, 3}, torch::kFloat32);
    torch::Tensor tensorR = torch::from_blob(imgR_norm.data, {1, imgR_norm.rows, imgR_norm.cols, 3}, torch::kFloat32);

    // 將張量從 HWC 轉換為 CHW
    tensorL = tensorL.permute({0, 3, 1, 2}).contiguous();
    tensorR = tensorR.permute({0, 3, 1, 2}).contiguous();

    // 計算填充大小，確保尺寸是 32 的倍數
    int h = tensorL.size(2);
    int w = tensorL.size(3);
    int h_pad = (32 - (h % 32)) % 32;
    int w_pad = (32 - (w % 32)) % 32;

    // 對張量進行填充
    tensorL = torch::nn::functional::pad(tensorL, torch::nn::functional::PadFuncOptions({0, w_pad, 0, h_pad}));
    tensorR = torch::nn::functional::pad(tensorR, torch::nn::functional::PadFuncOptions({0, w_pad, 0, h_pad}));

    // 將張量移至 CUDA
    tensorL = tensorL.to(torch::kCUDA);
    tensorR = tensorR.to(torch::kCUDA);

    // 通過模型進行推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensorL);
    inputs.push_back(tensorR);

    torch::Tensor output = module.forward(inputs).toTensor();

    // 輸出張量尺寸以進行檢查
    std::cout << "輸出張量尺寸: " << output.sizes() << std::endl;

    // 獲取 output 張量的維度數
    int output_dims = output.dim();

    // 根據 output 的維度數，調整切片操作
    if (output_dims == 4) {
        // output 的尺寸為 [N, C, H, W]
        output = output.slice(/*dim=*/2, /*start=*/0, /*end=*/h).slice(/*dim=*/3, /*start=*/0, /*end=*/w);
        // 壓縮維度
        output = output.squeeze(0).squeeze(0);
    }
    else if (output_dims == 3) {
        // output 的尺寸為 [N, H, W]
        output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/h).slice(/*dim=*/2, /*start=*/0, /*end=*/w);
        // 壓縮維度
        output = output.squeeze(0);
    }
    else {
        std::cerr << "意外的輸出張量維度數: " << output_dims << std::endl;
        return -1;
    }

    // 將輸出移回 CPU 並轉換為 OpenCV Mat
    output = output.detach().to(torch::kCPU);

    // 檢查輸出尺寸
    std::cout << "最終輸出張量尺寸: " << output.sizes() << std::endl;

    // 將張量轉換為 CV_32F Mat
    disp = cv::Mat(output.size(0), output.size(1), CV_32F, output.data_ptr<float>()).clone(); // 使用 clone() 防止資料被釋放

    // 對視差圖進行標準化以進行顯示
    cv::Mat disp_norm;
    cv::normalize(disp, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 應用彩色映射
    cv::Mat disp_color;
    cv::applyColorMap(disp_norm, disp_color, cv::COLORMAP_MAGMA);

    // 顯示視差圖
    cv::imshow("disp_norm", disp_norm);
    cv::imshow("disp_color", disp_color);
    // cv::imshow("disp", disp); // 如果需要顯示原始視差圖，可以取消註解

    // 設置鼠標回調函數
    cv::setMouseCallback("disp_norm", onMouse, nullptr);

    cv::waitKey(0);

    return 0;
}

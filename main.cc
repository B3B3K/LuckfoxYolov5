#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <vector>
#include <linux/i2c-dev.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "luckfox_mpi.h"
#include "yolov5.h"

#define DISP_WIDTH  720
#define DISP_HEIGHT 480
#define WEB_PORT 8080

struct AppConfig {
    bool show_help = false;
    bool enable_yolo = false;
    bool enable_web = false;
};

// Global variables for web server
int web_socket = -1;
pthread_t web_thread;
cv::Mat current_frame;
pthread_mutex_t frame_mutex = PTHREAD_MUTEX_INITIALIZER;
std::vector<object_detect_result> current_detections;

// disp size
int width = DISP_WIDTH;
int height = DISP_HEIGHT;

// model size
int model_width = 640;
int model_height = 640;	
float scale;
int leftPadding;
int topPadding;

cv::Mat letterbox(cv::Mat input) {
    float scaleX = (float)model_width / (float)width; 
    float scaleY = (float)model_height / (float)height; 
    scale = scaleX < scaleY ? scaleX : scaleY;
    
    int inputWidth = (int)((float)width * scale);
    int inputHeight = (int)((float)height * scale);

    leftPadding = (model_width - inputWidth) / 2;
    topPadding = (model_height - inputHeight) / 2;    
    
    cv::Mat inputScale;
    cv::resize(input, inputScale, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_LINEAR);	
    cv::Mat letterboxImage(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect roi(leftPadding, topPadding, inputWidth, inputHeight);
    inputScale.copyTo(letterboxImage(roi));

    return letterboxImage; 	
}

void parse_arguments(int argc, char* argv[], AppConfig &config) {
    static struct option long_options[] = {
        {"yolo", no_argument, 0, 'y'},
        {"web", no_argument, 0, 'w'},
        {"help", no_argument, 0, '?'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "yw?", long_options, NULL)) != -1) {
        switch (opt) {
            case 'y': config.enable_yolo = true; break;
            case 'w': config.enable_web = true; break;
            case '?': config.show_help = true; break;
        }
    }
}

void print_help(const char* prog_name) {
    printf("Usage: %s [options]\nOptions:\n"
           "  -y, --yolo              Enable YOLO object detection\n"
           "  -w, --web               Enable web server for live preview\n"
           "  -?, --help              Show this help\n", prog_name);
}

void* web_server(void* arg) {
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[1024];
    char response_header[512];
    char image_buffer[100 * 1024]; // Enough for JPEG

    // Create socket
    if ((web_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket failed");
        return NULL;
    }

    // Set socket options
    int opt = 1;
    if (setsockopt(web_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        close(web_socket);
        return NULL;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(WEB_PORT);

    // Bind socket
    if (bind(web_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(web_socket);
        return NULL;
    }

    // Listen
    if (listen(web_socket, 3) < 0) {
        perror("listen");
        close(web_socket);
        return NULL;
    }

    printf("Web server running on port %d\n", WEB_PORT);

    while (1) {
        int client_socket = accept(web_socket, (struct sockaddr *)&client_addr, &client_len);
        if (client_socket < 0) {
            perror("accept");
            continue;
        }

        // Read HTTP request
        read(client_socket, buffer, sizeof(buffer));

        // Check if it's a request for the image
        if (strstr(buffer, "GET /image.jpg") != NULL) {
            pthread_mutex_lock(&frame_mutex);
            
            // Convert frame to JPEG
            std::vector<uchar> jpeg_buffer;
            std::vector<int> params;
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(80);
            
            cv::Mat display_frame;
            current_frame.copyTo(display_frame);
            
            // Draw detections if available
            if (!current_detections.empty()) {
                for (const auto& det : current_detections) {
                    cv::rectangle(display_frame, 
                        cv::Point(det.box.left, det.box.top),
                        cv::Point(det.box.right, det.box.bottom),
                        cv::Scalar(0, 255, 0), 2);
                    
                    char label[128];
                    snprintf(label, sizeof(label), "%s %.1f%%", 
                            coco_cls_to_name(det.cls_id), det.prop * 100);
                    
                    cv::putText(display_frame, label, 
                               cv::Point(det.box.left, det.box.top - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                               cv::Scalar(0, 255, 0), 1);
                }
            }
            
            cv::imencode(".jpg", display_frame, jpeg_buffer, params);
            pthread_mutex_unlock(&frame_mutex);

            // Prepare HTTP response
            int content_length = jpeg_buffer.size();
            snprintf(response_header, sizeof(response_header),
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: image/jpeg\r\n"
                    "Content-Length: %d\r\n"
                    "Connection: close\r\n"
                    "\r\n", content_length);

            // Send header and image data
            write(client_socket, response_header, strlen(response_header));
            write(client_socket, jpeg_buffer.data(), jpeg_buffer.size());
        }
        else {
            // Serve HTML page
            const char *html_response = 
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "Connection: close\r\n"
                "\r\n"
                "<html><head><title>Object Detection</title></head><body>"
                "<h1>Live Object Detection</h1>"
                "<img src='/image.jpg' style='width:640px;'/>"
                "<script>"
                "setInterval(function(){"
                "  document.querySelector('img').src = '/image.jpg?' + Date.now();"
                "}, 100);"
                "</script>"
                "</body></html>";

            write(client_socket, html_response, strlen(html_response));
        }

        close(client_socket);
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    AppConfig config;
    parse_arguments(argc, argv, config);
    if (config.show_help) { 
        print_help(argv[0]); 
        return 0; 
    }

    // Start web server if enabled
    if (config.enable_web) {
        if (pthread_create(&web_thread, NULL, web_server, NULL) != 0) {
            perror("Failed to start web server");
            config.enable_web = false;
        }
    }

    // Initialize YOLO if enabled
    rknn_app_context_t rknn_app_ctx;
    object_detect_result_list od_results;
    if (config.enable_yolo) {
        const char *model_path = "./model/yolov5.rknn";
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));    
        if (init_yolov5_model(model_path, &rknn_app_ctx) != 0) {
            fprintf(stderr, "Failed to initialize YOLO model\n");
            config.enable_yolo = false;
        } else {
            printf("YOLO model initialized successfully\n");
            init_post_process();
        }
    }

    // Initialize camera system
    system("RkLunch-stop.sh");
    RK_S32 s32Ret = 0; 
    int sX,sY,eX,eY; 
    char text[16];

    RK_BOOL multi_sensor = RK_FALSE;    
    const char *iq_dir = "/etc/iqfiles";
    rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
    SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
    SAMPLE_COMM_ISP_Run(0);

    // rkmpi init
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        RK_LOGE("rk mpi sys init fail!");
        return -1;
    }

    // Create Pool
    MB_POOL_CONFIG_S PoolCfg;
    memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
    PoolCfg.u64MBSize = width * height * 3;
    PoolCfg.u32MBCnt = 1;
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
    MB_POOL src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    printf("Create Pool success !\n");    

    // Get MB from Pool 
    MB_BLK src_Blk = RK_MPI_MB_GetMB(src_Pool, width * height * 3, RK_TRUE);
    
    // Build frame buffer
    VIDEO_FRAME_INFO_S h264_frame;
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = width;
    h264_frame.stVFrame.u32VirHeight = height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_RGB888; 
    h264_frame.stVFrame.u32FrameFlag = 160;
    h264_frame.stVFrame.pMbBlk = src_Blk;
    unsigned char *data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    cv::Mat frame(cv::Size(width, height), CV_8UC3, data);

    // vi init
    if (vi_dev_init() != 0) {
        fprintf(stderr, "Failed to initialize VI device\n");
        return -1;
    }
    if (vi_chn_init(0, width, height) != 0) {
        fprintf(stderr, "Failed to initialize VI channel\n");
        return -1;
    }
    printf("Camera system initialized\n");

    // Main processing loop
    while (1) {
        // Get video frame
        VIDEO_FRAME_INFO_S stViFrame;
        s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
        if(s32Ret == RK_SUCCESS) {
            void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);    

            cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
            cv::Mat bgr(height, width, CV_8UC3, data);            
            
            cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);
            cv::resize(bgr, frame, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            
            if (config.enable_web) {
                pthread_mutex_lock(&frame_mutex);
                frame.copyTo(current_frame);
                pthread_mutex_unlock(&frame_mutex);
            }
            
            if (config.enable_yolo) {
                cv::Mat letterboxImage = letterbox(frame);    
                memcpy(rknn_app_ctx.input_mems[0]->virt_addr, letterboxImage.data, model_width*model_height*3);        
                inference_yolov5_model(&rknn_app_ctx, &od_results);

                // Store detections for web display
                if (config.enable_web) {
                    pthread_mutex_lock(&frame_mutex);
                    current_detections.clear();
                    for (int i = 0; i < od_results.count; i++) {
                        current_detections.push_back(od_results.results[i]);
                    }
                    pthread_mutex_unlock(&frame_mutex);
                }

                for(int i = 0; i < od_results.count; i++) {
                    if(od_results.count >= 1) { 
                        const object_detect_result *det_result = &(od_results.results[i]);
                        sX = (int)(det_result->box.left);   
                        sY = (int)(det_result->box.top);     
                        eX = (int)(det_result->box.right);   
                        eY = (int)(det_result->box.bottom);
                        
                        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                                              sX, sY, eX, eY, det_result->prop);
                    }
                }
            }

            // Release frame 
            s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
            if (s32Ret != RK_SUCCESS) {
                RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
            }
        }
    }

    // Cleanup
    RK_MPI_MB_ReleaseMB(src_Blk);
    RK_MPI_MB_DestroyPool(src_Pool);
    
    RK_MPI_VI_DisableChn(0, 0);
    RK_MPI_VI_DisableDev(0);

    SAMPLE_COMM_ISP_Stop(0);
    RK_MPI_SYS_Exit();

    if (config.enable_yolo) {
        release_yolov5_model(&rknn_app_ctx);        
        deinit_post_process();
    }

    if (config.enable_web) {
        close(web_socket);
        pthread_join(web_thread, NULL);
    }

    return 0;
}

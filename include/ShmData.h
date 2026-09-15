/**
 * 共享内存数据结构 — SLAM 进程与 Qt 界面进程的通信桥梁
 * 纯 C 风格 POD，可安全地放入共享内存
 */

#ifndef SHM_DATA_H
#define SHM_DATA_H

#include <stdint.h>

// 共享内存名称
#define SHM_NAME_CTRL   "/slam_shm_ctrl"
#define SHM_NAME_FRAME  "/slam_shm_frame"
#define SHM_NAME_MAP    "/slam_shm_map"
#define SEM_NAME_FRAME  "/slam_sem_frame"

// 图像最大尺寸
#define SHM_MAX_IMG_W   1920
#define SHM_MAX_IMG_H   1080
#define SHM_MAX_FRAME_BYTES  (SHM_MAX_IMG_W * SHM_MAX_IMG_H * 3)

// ============================================================
// 控制/状态段 (小，频繁读写)
// ============================================================
struct ShmCtrl {
    // ── Qt → SLAM 命令 ──
    // 值：0=无操作, 1=激活/开启, -1=停用/关闭
    volatile int32_t cmd_localization;
    volatile int32_t cmd_step_by_step;
    volatile int32_t cmd_reset;         // 1=重置地图
    volatile int32_t cmd_shutdown;      // 1=关闭 SLAM 进程
    volatile int32_t cmd_step;          // 1=步进一帧 (步进模式下)
    volatile int32_t cmd_vis_mode;      // -1=原图(短焦), 1=动态一致性, 2=长短焦(右目长焦)

    // 绘制选项 (由 Qt 设置，SLAM 读取后用于 MapDrawer)
    volatile int32_t draw_points;
    volatile int32_t draw_keyframes;
    volatile int32_t draw_graph;
    volatile int32_t draw_plane;
    volatile int32_t draw_3dbox;
    volatile int32_t highlight_lba;

    // ── SLAM → Qt 状态 ──
    volatile int32_t frame_counter;     // 每完成一帧 +1
    volatile int32_t slam_ready;        // 0=未初始化, 1=已就绪
    volatile int32_t tracking_state;    // 0=未初始化, 1=OK, 2=丢失
    volatile int32_t map_points_cnt;
    volatile int32_t keyframes_cnt;

    // 相机位姿 4x4 列主序 (T_wc)
    float camera_pose[16];

    // 当前帧图像的实际尺寸
    int32_t image_width;
    int32_t image_height;

    // SLAM 处理完命令后回写到 ack 字段
    volatile int32_t ack_localization;  // 处理后回写 cmd_localization 的值
    volatile int32_t ack_step_by_step;

    char _pad[28];  // 对齐到 128 字节边界
};

// ============================================================
// 图像帧段 (大，每帧覆写)
// ============================================================
struct ShmFrame {
    volatile int32_t frame_id;              // 与 ShmCtrl.frame_counter 同步
    unsigned char  data[SHM_MAX_FRAME_BYTES]; // RGB 格式图像
};

// ============================================================
// 3D 地图数据段 (轨迹 + 关键帧位姿)
// ============================================================
#define MAX_TRAJ_POINTS  2000
#define MAX_KF_POSES      500
#define MAX_MAP_POINTS    3000
#define MAX_3D_BOXES       100

struct ShmMap {
    volatile int32_t map_frame_id;       // 与 ShmCtrl.frame_counter 同步

    // ── 相机轨迹 (环形缓冲) ──
    volatile int32_t traj_count;         // 有效条目数
    volatile int32_t traj_start;         // 环形缓冲起始索引
    float           traj[MAX_TRAJ_POINTS][3];  // x,y,z

    // ── 关键帧位姿 + 状态 ──
    // kf_status: 0=普通, 1=首帧, 2=固定, 3=LBA优化
    volatile int32_t kf_count;           // 有效关键帧数
    float           kf_poses[MAX_KF_POSES][16]; // 4x4 列主序 T_wc
    uint8_t         kf_status[MAX_KF_POSES];    // 关键帧状态标记

    // ── 地图点 + 颜色 ──
    volatile int32_t mp_count;           // 有效地图点数
    float           map_points[MAX_MAP_POINTS][3];  // x,y,z
    // mp_colors: RGB 各 1 字节, 由 SLAM 端填入
    // 约定: (0,0,0)=黑色=未知/历史点 (由Qt端按需覆盖渲染)
    uint8_t         mp_colors[MAX_MAP_POINTS][3];  // RGB 颜色

    // ── 3D 检测框 + 平面法向量 ──
    volatile int32_t box_count;              // 有效3D框数
    float            boxes[MAX_3D_BOXES][11]; // [0-2]=center, [3]=width, [4]=depth, [5]=height,
                                              // [6]=class_id, [7]=nObs, [8-10]=heading(车长轴,平面内)
    float            plane_normal[3];         // 地面平面法向量
    float            plane_offset;            // 地面平面偏移 d (n·P = d)

    char _pad[32];
};

// 编译期检查对齐
static_assert(sizeof(ShmCtrl)  <= 4096,  "ShmCtrl too large");
static_assert(sizeof(ShmFrame) <= SHM_MAX_FRAME_BYTES + 64, "ShmFrame too large");

#endif // SHM_DATA_H

const std::vector<std::string> CLASS_NAMES = {
    "pedestrian",   "people",   "bicycle",  "car",  "van",
    "truck",    "tricycle", "awning-tricycle",  "bus",  "motor" };


const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
};


const std::vector<float> MOTION_PROBABILITIES = {
    0.4f,  // pedestrian - 行人
    0.4f,  // people - 人群
    0.4f,  // bicycle - 自行车
    0.4f,  // car - 汽车
    0.4f,  // van - 货车
    0.4f,  // truck - 卡车
    0.4f,  // tricycle - 三轮车
    0.4f,  // awning-tricycle - 带篷三轮车
    0.4f,  // bus - 公交车
    0.4f   // motor - 摩托车
};

// 检测框边距扩展（像素），用于特征点匹配时的语义区域判定
// class_id: 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
//           5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
const std::vector<int> SEMANTIC_BOX_MARGINS = {
    2,  // pedestrian - 行人很小
    2,  // people - 人群
    2,  // bicycle - 自行车
    3,  // car - 汽车
    4,  // van - 货车
    5,  // truck - 卡车
    3,  // tricycle - 三轮车
    3,  // awning-tricycle - 带篷三轮车
    5,  // bus - 公交车
    3   // motor - 摩托车
};

const std::vector<float> THRESHOLD_RATIOS = {
    1.20f,  // pedestrian - 行人
    0.8f,  // people - 人群
    1.0f,  // bicycle - 自行车
    0.9f,  // car - 汽车
    1.0f,  // van - 货车
    1.0f,  // truck - 卡车
    1.0f,  // tricycle - 三轮车
    1.0f,  // awning-tricycle - 带篷三轮车
    1.2f,  // bus - 公交车
    1.0f   // motor - 摩托车
};
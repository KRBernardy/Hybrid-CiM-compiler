#include "common.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ConfigGenerator {
    private:
        ModelImpl *model_;
        Placer *placer_;

        json configGen();

    public:
        ConfigGenerator(ModelImpl* model, Placer* placer);
};
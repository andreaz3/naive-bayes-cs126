#include <core/Image.h>
#include <map>
#include <vector>


using namespace std;
using std::ifstream;
using std::string;

namespace naivebayes {

    Image::Image(const size_t &label, const vector<vector<size_t>> &image_shading) {
        this->label = label;
        this->image_shading = image_shading;
    }

    size_t Image::GetLabel() {
        return label;
    }

    vector<vector<size_t>> Image::GetImageShading() {
        return image_shading;
    }

}  // namespace naivebayes
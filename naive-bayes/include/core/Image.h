#include <string>
#include <iostream>
#include <vector>
#include <map>

#pragma once

using namespace std;

namespace naivebayes {
    class Image {
    private:
        size_t label;
        vector<vector<size_t>> image_shading;

    public:
        /**
         * Constructor for an image, which takes in a label and image shading.
         * @param label
         * @param image_shading
         */
        Image(const size_t &label, const vector<vector<size_t>> &image_shading);

        /**
         * Getter for Image.
         * @return
         */
        size_t GetLabel();

        /**
         * Getter for image shading.
         * @return
         */
        vector<vector<size_t>> GetImageShading();

    };

}  // namespace naivebayes
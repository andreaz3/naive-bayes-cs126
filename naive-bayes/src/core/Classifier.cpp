#include <fstream>
#include <vector>
#include <algorithm>
#include "core/Classifier.h"
#include <cmath>

using namespace std;

namespace naivebayes {

    double Classifier::ReturnAccuracy(Model &trained_model, Model &testing_model) {
        double correct_labels = 0;
        for (Image &image : testing_model.image_vector) {
            vector<vector<size_t>> curr_image_shading = image.GetImageShading();
            if (CalculateLikeliestLabel(trained_model, curr_image_shading) == image.GetLabel()) {
                correct_labels++;
            } else {
                continue;
            }
        }
        double classifier_accuracy = correct_labels/testing_model.image_vector.size();
        return classifier_accuracy;
    }

    size_t Classifier::CalculateLikeliestLabel(Model &trained_model, vector<vector<size_t>> &image_shading) {
        vector<double> probability_of_labels;
        for (size_t k = 0; k < trained_model.label_options; k++) {
            double num_to_add = log(trained_model.all_priors[k]);
            for (size_t i = 0; i < trained_model.image_x_dimension; i++) {
                for (size_t j = 0; j < trained_model.image_y_dimension; j++) {
                    size_t curr_shade = image_shading[i][j];
                    num_to_add += log(trained_model.all_likelihoods[i][j][curr_shade][k]);
                }
            }
            probability_of_labels.push_back(num_to_add);
        }
        return ReturnLikeliestLabel(probability_of_labels);
    }

    size_t Classifier::ReturnLikeliestLabel(vector<double> &probability_of_labels) {
        double max_value = probability_of_labels.at(0);
        size_t max_label = 0;
        for (size_t i = 0; i < probability_of_labels.size(); i++) {
            if (probability_of_labels.at(i) >= max_value) {
                max_value = probability_of_labels.at(i);
                max_label = i;
            }
        }
        return max_label;
    }
}

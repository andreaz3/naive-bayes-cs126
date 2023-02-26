#include "Model.h"

#pragma once

namespace naivebayes {

    class Classifier {

    private:
    public:
        /**
         * Returns the accuracy of a testing model given the trained model.
         * @param trained_model
         * @param testing_model
         * @return the ratio of correct labels over total labels.
         */
        double ReturnAccuracy(Model &trained_model, Model &testing_model);

        /**
         * Calculates the likeliest label given the trained model and the image_shading.
         * of an image in the testing model.
         * @param trained_model
         * @param image_shading
         * @return the likeliest label.
         */
        size_t CalculateLikeliestLabel(Model &trained_model, vector<vector<size_t>> &image_shading);

        /**
         * Returns the likeliest label given the probability of all labels.
         * Parses through each element and finds the index of the max.
         * @param probability_of_labels
         * @return the label with the highest probability.
         */
        size_t ReturnLikeliestLabel(vector<double> &probability_of_labels);
    };
}



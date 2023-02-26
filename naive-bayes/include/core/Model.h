#include <string>
#include "core/Image.h"
#include <vector>

#pragma once

using namespace std;

namespace naivebayes {
    class Model {
    private:
        double k_laplace_smoothing = 1;
    public:
        Model();
        /**
         * Constructor for method passed an image vector from processor.
         * @param image_vector
         */
        explicit Model(vector<Image> &image_vector);

        /**
         * A vector of all training images.
         */
        vector<Image> image_vector;

        /**
         * A vector of all the image shadings, in the order of labels
         */
        vector<vector<vector<vector<size_t>>>> ordered_trained_image_shadings;


        /**
         * Dimensions and label & shading options for the images.
         */
        size_t image_x_dimension = 28;
        size_t image_y_dimension = 28;
        size_t shading_options = 2;
        size_t label_options = 10;

        /**
         * A vector of all the calculated priors.
         */
        vector<double> all_priors;

        /**
         * A vector of all the calculated likelihoods.
         */
        vector<vector<vector<vector<double>>>> all_likelihoods;

        /**
         * Method called to populate the all image label vector
         * and all image shading vector using image vector
         */
        void PopulateTrainedImageShadingVector();

        /**
         * Calculates the prior given a label
         * @param label
         * @return the prior calculation (probability)
         */
        double PriorCalculation(const size_t &label);

        /**
         * calculates the likelihood given the following:
         * @param x_index
         * @param y_index
         * @param shading
         * @param label
         * @return  the likelihood calculation (probability)
         */
        double LikelihoodCalculation(const size_t &x_index,
                                      const size_t &y_index,
                                      const size_t &shading,
                                      const size_t &label);

        /**
         * Populates the prior vector and likelihood vector using the
         * PriorCalculation method and LikelihoodCalculation method.
         */
        void TrainImages();

        /**
         * Overloaded Operator of << to save a trained model.
         * @param output
         * @param model
         * @return output
         */
        friend ostream &operator<<(ostream &output, Model &model);

        /**
         * Overloaded Operator of >> to read a trained model.
         * @param input
         * @param model
         * @return input
         */
        friend istream &operator>>(istream &input, Model &model);

    };
}

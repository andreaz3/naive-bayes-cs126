#include "core/Model.h"
#include "core/Image.h"
#include <fstream>

namespace naivebayes {

    Model::Model() {

    }

    Model::Model(vector<Image> &image_vector) {
        this->image_vector = image_vector;
    }

    void Model::PopulateTrainedImageShadingVector(){
        ordered_trained_image_shadings.resize(label_options);
        for (Image image : image_vector) {
            switch(image.GetLabel()) {
                case 0: ordered_trained_image_shadings[0].push_back(image.GetImageShading());
                    break;
                case 1: ordered_trained_image_shadings[1].push_back(image.GetImageShading());
                    break;
                case 2: ordered_trained_image_shadings[2].push_back(image.GetImageShading());
                    break;
                case 3: ordered_trained_image_shadings[3].push_back(image.GetImageShading());
                    break;
                case 4: ordered_trained_image_shadings[4].push_back(image.GetImageShading());
                    break;
                case 5: ordered_trained_image_shadings[5].push_back(image.GetImageShading());
                    break;
                case 6: ordered_trained_image_shadings[6].push_back(image.GetImageShading());
                    break;
                case 7: ordered_trained_image_shadings[7].push_back(image.GetImageShading());
                    break;
                case 8: ordered_trained_image_shadings[8].push_back(image.GetImageShading());
                    break;
                case 9: ordered_trained_image_shadings[9].push_back(image.GetImageShading());
                    break;
            }
        }
    }

    double Model::PriorCalculation(const size_t &label) {

        size_t target_image_shadings_size = ordered_trained_image_shadings.at(label).size();

        double prior = (k_laplace_smoothing + target_image_shadings_size)
                / (label_options * k_laplace_smoothing + image_vector.size());
        return prior;
    }

    double Model::LikelihoodCalculation(const size_t &x_index,
                                      const size_t &y_index,
                                      const size_t &shading,
                                      const size_t &label) {

        if (x_index >= image_x_dimension || y_index >= image_y_dimension) {
            throw invalid_argument("that index doesn't exist in the image");
        }

        size_t target_image_shadings_size = ordered_trained_image_shadings.at(label).size();
        size_t num_target_images = 0;

        for (size_t i = 0; i < target_image_shadings_size; i++) {
            if (ordered_trained_image_shadings.at(label)[i][x_index][y_index] == shading) {
                num_target_images++;
            }
        }

        double likelihood = (k_laplace_smoothing + num_target_images) /
                          (shading_options * k_laplace_smoothing + target_image_shadings_size);

        return likelihood;

    }

    void Model::TrainImages() {

        vector<size_t> all_image_label_options{0,1,2,3,4,5,6,7,8,9};

        for (size_t &image_label : all_image_label_options) {
            all_priors.push_back(PriorCalculation(image_label));
        }

        all_likelihoods.resize(image_x_dimension);
        for (size_t j = 0; j < image_x_dimension; j++) {
            all_likelihoods[j].resize(image_y_dimension);
            for (size_t k = 0; k < image_y_dimension; k++) {
                all_likelihoods[j][k].resize(shading_options);
                for (size_t l = 0; l < shading_options; l++) {
                    all_likelihoods[j][k][l].resize(label_options);
                    for (size_t m = 0; m < label_options; m++) {
                        all_likelihoods[j][k][l][m] = LikelihoodCalculation(j,k,l,m);
                    }
                }
            }
        }
    }

    ostream &operator<<(ostream &output, naivebayes::Model &model) {

        for (double &prior : model.all_priors) {
            output << '*' << prior << '\n';
        }

        for (size_t j = 0; j < model.image_x_dimension; j++) {
            for (size_t k = 0; k < model.image_y_dimension; k++) {
                for (size_t l = 0; l < model.shading_options; l++) {
                    for (size_t m = 0; m < model.label_options; m++) {
                        output <<'*'<<j<<","<<k<<","<<l<<m << model.all_likelihoods[j][k][l][m] << '\n';
                    }
                }
            }
        }

        return output;
    }


    istream &operator>>(istream &input, naivebayes::Model &model) {

        string line;

        //for loading a model
        string prior_str;
        string likelihood_str;
        double prior;
        double likelihood;

        size_t label_counter = 0;

        //for converting training data to images
        size_t label;
        size_t counter = 0;
        vector<vector<size_t>> image_shading_vec(model.image_x_dimension,
                                                 vector<size_t>(model.image_y_dimension));

        while(getline(input, line)) {
            if (line.empty()) {
                break;
            }
            if (line.at(0) == '*') {
                if (label_counter < model.label_options) {
                    for (char &character : line.substr(1, line.size() - 1)) {
                        prior_str += character;
                    }
                    prior = stod(prior_str);
                    model.all_priors.push_back(prior);
                    prior_str = "";
                    label_counter++;
                } else {
                    //Getting indices
                    size_t x_index;
                    size_t x_counter;
                    size_t y_index;
                    size_t y_counter;
                    size_t shade_index;
                    size_t label_index;
                    for (int i = 1; line.at(i) != ','; i++) {
                        if (i > 1) {
                            x_index = stoi(line.substr(1,1 + i));
                        } else {
                            char char_x = line.at(i);
                            x_index = char_x - '0';
                        }
                        x_counter = i + 2;
                    }
                    for (int i = x_counter; line.at(i) != ','; i++) {
                        if (i > x_counter) {
                            y_index = stoi(line.substr(x_counter,x_counter + i));
                        } else {
                            char char_y = line.at(i);
                            y_index = char_y - '0';
                        }
                        y_counter = i + 2;
                    }

                    char char_shade = line.at(y_counter);
                    shade_index = char_shade - '0';
                    char char_label = line.at(y_counter + 1);
                    label_index = char_label - '0';

                    //getting likelihood
                    for (char &character : line.substr(y_counter + 2, line.size() - 1)) {
                        likelihood_str += character;
                    }
                    likelihood = stod(likelihood_str);

                    //resizing 4d vector
                    model.all_likelihoods.resize(model.image_x_dimension);
                    model.all_likelihoods[x_index].resize(model.image_y_dimension);
                    model.all_likelihoods[x_index][y_index].resize(model.shading_options);
                    model.all_likelihoods[x_index][y_index][shade_index].resize(model.label_options);

                    //putting the likelihood numbers into the likelihood vector
                    model.all_likelihoods [x_index][y_index][shade_index][label_index] = likelihood;
                    likelihood_str= "";
                }
            } else {
                if (counter < model.image_x_dimension) {
                    counter++;

                    if (counter == 1) {
                        char current_char = line.at(0);
                        label = current_char - '0';
                        //model.trained_image_labels.push_back(label);
                    } else {
                        for (size_t i = 0; i < model.image_y_dimension; i++) {
                            char current_char = line.at(i);
                            if (current_char == ' ') {
                                size_t unshaded = 0;
                                image_shading_vec[counter - 2][i] = unshaded;
                            } else if (current_char == '+' || current_char == '#') {
                                size_t shaded = 1;
                                image_shading_vec[counter - 2][i] = shaded;
                            }
                        }
                        //model.trained_image_shadings.push_back(image_shading_vec);
                    }
                } else {
                    model.image_vector.emplace_back(label, image_shading_vec);
                    counter = 0;
                }
            }
        }
        return input;
    }

}
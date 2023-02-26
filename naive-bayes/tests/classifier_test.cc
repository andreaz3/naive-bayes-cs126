#include <catch2/catch.hpp>

#include <core/Image.h>
#include <core/Model.h>
#include <core/Classifier.h>
#include <fstream>

using naivebayes::Model;
using naivebayes::Classifier;
using naivebayes::Image;

TEST_CASE("Testing ReturnLikeliestLabel") {
    SECTION("Max at 0 index") {
        vector<double> random_order{0.98,0.1,0.2,0.3,0.14};
        Classifier classifier;
        size_t actual = classifier.ReturnLikeliestLabel(random_order);
        REQUIRE(actual == 0);
    }
    SECTION("Max at last index") {
        vector<double> random_order{99.2,0.0001,57,88.2,100.0001, 100.0002};
        Classifier classifier;
        size_t actual = classifier.ReturnLikeliestLabel(random_order);
        REQUIRE(actual == 5);
    }
    SECTION("Max at random index") {
        vector<double> random_order{0.3,9.1,2.2,0,-0.1};
        Classifier classifier;
        size_t actual = classifier.ReturnLikeliestLabel(random_order);
        REQUIRE(actual == 1);
    }
}

TEST_CASE("Testing ReturnAccuracy") {
    SECTION("By training images") {
        Model trained_model;
        std::ifstream train_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/training_images.txt");
        std::istream& stream_train = train_images_file;
        stream_train >> trained_model;
        trained_model.PopulateTrainedImageShadingVector();
        trained_model.TrainImages();

        Model testing_model;
        std::ifstream testing_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/testimagesandlabels.txt");
        std::istream& stream_test = testing_images_file;
        stream_test >> testing_model;

        Classifier classifier;
        REQUIRE(classifier.ReturnAccuracy(trained_model, testing_model) >= 0.7);
    }
    SECTION("By already trained model") {
        naivebayes::Model trained_model;
        std::ifstream trained_model_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/empty");
        std::istream& stream2 = trained_model_file;
        stream2 >> trained_model;

        Model testing_model;
        std::ifstream testing_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/testimagesandlabels.txt");
        std::istream& stream_test = testing_images_file;
        stream_test >> testing_model;

        Classifier classifier;
        REQUIRE(classifier.ReturnAccuracy(trained_model, testing_model) >= 0.7);
    }
}

TEST_CASE("Testing CalculateLikeliestLabel") {
    SECTION("By training images") {
        Model trained_model;
        std::ifstream train_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/training_images.txt");
        std::istream& stream_train = train_images_file;
        stream_train >> trained_model;
        trained_model.PopulateTrainedImageShadingVector();
        trained_model.TrainImages();

        Model testing_model;
        std::ifstream testing_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/testimagesandlabels.txt");
        std::istream& stream_test = testing_images_file;
        stream_test >> testing_model;

        Classifier classifier;
        vector<vector<size_t>> image_shading = testing_model.image_vector.at(10).GetImageShading();
        size_t num = classifier.CalculateLikeliestLabel(trained_model, image_shading);
        REQUIRE(num == 4);
    }
    SECTION("By already trained model") {
        naivebayes::Model trained_model;
        std::ifstream trained_model_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/empty");
        std::istream& stream2 = trained_model_file;
        stream2 >> trained_model;

        Model testing_model;
        std::ifstream testing_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/testimagesandlabels.txt");
        std::istream& stream_test = testing_images_file;
        stream_test >> testing_model;

        Classifier classifier;
        vector<vector<size_t>> image_shading = testing_model.image_vector.at(9).GetImageShading();
        size_t num = classifier.CalculateLikeliestLabel(trained_model, image_shading);
        REQUIRE(num == 0);
    }
}
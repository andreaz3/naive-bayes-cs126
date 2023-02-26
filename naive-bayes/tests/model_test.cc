#include <catch2/catch.hpp>

#include <core/Image.h>
#include <core/Model.h>
#include <core/Classifier.h>
#include <fstream>

using naivebayes::Model;
using naivebayes::Classifier;
using naivebayes::Image;

TEST_CASE("Test Simple_Training_Images") {
    naivebayes::Model model;
    std::ifstream train_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/simple_training_images.txt");
    std::istream& stream = train_images_file;
    stream >> model;

    SECTION("Testing constructor") {
        REQUIRE(model.image_vector.size() == 3);
    }

    SECTION("Testing priors of labels that exist in training data") {
        REQUIRE(model.PriorCalculation(5) == Approx(0.154).margin(.01));
        REQUIRE(model.PriorCalculation(0) == Approx(0.154).margin(.01));
        REQUIRE(model.PriorCalculation(4) == Approx(0.154).margin(.01));
    }
    SECTION("Testing priors of labels that don't exist in training data") {
        REQUIRE(model.PriorCalculation(2) == Approx(0.0769).margin(.01));
        REQUIRE(model.PriorCalculation(19) == Approx(0.0769).margin(.01));
    }
    SECTION("Testing Likelihood Calculations") {
        //Will be tested in depth below...
    }

    model.TrainImages();
    SECTION("Testing priors getting added to all_priors correctly") {
        REQUIRE(model.all_priors.size() == 3);
    }
    SECTION("Testing likelihoods getting added to all_likelihoods correctly") {
        REQUIRE(model.all_likelihoods.size() == 28);
        REQUIRE(model.all_likelihoods[1].size() == 28);
        REQUIRE(model.all_likelihoods[1][1].size() == 2);
        REQUIRE(model.all_likelihoods[1][1][1].size() == 10);
    }
}

TEST_CASE("TESTING LIKELIHOODS") {
    naivebayes::Model model;
    std::ifstream train_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/simple_training_images.txt");
    std::istream& stream = train_images_file;
    stream >> model;
    model.PopulateTrainedImageShadingVector();

    SECTION("Index not applicable") {
        REQUIRE_THROWS_AS(model.LikelihoodCalculation(29,29,0,0), std::invalid_argument);
    }

    SECTION("Index with shading in a label that does exist, tested for shading") {
        REQUIRE(model.LikelihoodCalculation(9,12,1,5) == Approx(0.667).margin(.01));
        REQUIRE(model.LikelihoodCalculation(9,13,1,0) == Approx(0.667).margin(.01));
        REQUIRE(model.LikelihoodCalculation(9,5,1,4) == Approx(0.667).margin(.01));
    }
    SECTION("Index with shading in a label that does exist, tested for no shading") {
        REQUIRE(model.LikelihoodCalculation(9,12,0,5) == Approx(0.333).margin(.01));
        REQUIRE(model.LikelihoodCalculation(9,13,0,0) == Approx(0.333).margin(.01));
        REQUIRE(model.LikelihoodCalculation(9,5,0,4) == Approx(0.333).margin(.01));
    }
    SECTION("Index without shading in a label that does exist, tested for no shading") {
        REQUIRE(model.LikelihoodCalculation(0,0,0,5) == Approx(0.667).margin(.01));
        REQUIRE(model.LikelihoodCalculation(27,27,0,0) == Approx(0.667).margin(.01));
        REQUIRE(model.LikelihoodCalculation(1,1,0,4) == Approx(0.667).margin(.01));
    }
    SECTION("Index without shading in a label that does exist, tested for shading") {
        REQUIRE(model.LikelihoodCalculation(0,0,1,5) == Approx(0.333).margin(.01));
        REQUIRE(model.LikelihoodCalculation(27,27,1,0) == Approx(0.333).margin(.01));
        REQUIRE(model.LikelihoodCalculation(1,1,1,4) == Approx(0.333).margin(.01));
    }
    SECTION("Index with shading in a label that does not exist") {
        REQUIRE(model.LikelihoodCalculation(9,12,0,2) == Approx(0.5).margin(.01));
    }
    SECTION("Index without shading in a label that does not exist") {
        REQUIRE(model.LikelihoodCalculation(9,12,1,2) == Approx(0.5).margin(.01));

    }
}
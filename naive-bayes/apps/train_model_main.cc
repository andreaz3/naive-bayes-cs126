#include <iostream>

#include <core/Image.h>
#include <core/Classifier.h>
#include <core/Model.h>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {

    //reads training data
    naivebayes::Model model;
    std::ifstream train_images_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/training_images.txt");
    std::istream& stream = train_images_file;
    stream >> model;

    //trains model
    model.PopulateTrainedImageShadingVector();
    model.TrainImages();

    //saves trained model to a file
    std::ofstream myfile;
    myfile.open ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/empty");
    std::ostream& stream1 = myfile;
    stream1 << model << endl;
    myfile.close();

    //loads trained model file into a model
    naivebayes::Model model1;
    std::ifstream trained_model_file ("/Users/andreazhou/Downloads/Cinder/my-projects/naive-bayes-andreaz3/data/empty");
    std::istream& stream2 = trained_model_file;
    stream2 >> model1;

  return 0;
}

/**
  \ingroup TensorflowCppWrapper
  \file    image_example.cpp
  \brief   This image_example.cpp file contains the example of using the Tensorflow API for image prediction
  \author  kovalenko
  \date    2020-03-05

  Copyright:
  2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
  The copyright of this software source code is the property of HHI.
  This software may be used and/or copied only with the written permission
  of HHI and in accordance with the terms and conditions stipulated
  in the agreement/contract under which the software has been supplied.
  The software distributed under this license is distributed on an "AS IS" basis,
  WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
*/

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>

#include "tf_image.hpp"


void runSavedModel(cv::Mat im);
void run_single_image_example(cv::Mat image);
void run_expecting_image_example();
cv::Mat createHeatmap( const cv::Mat& heatmaps );

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main( ) {  

  std::cout << "Running C API loadSavedModel example:" << std::endl;
  cv::Mat tNFImage = cv::imread("/home/kostasl/workspace/hello_tf_c_api/build/nonfish_sample.jpg", cv::IMREAD_UNCHANGED );
  cv::Mat tNFImageB = cv::imread("/home/kostasl/workspace/zebrafishtrack/tensorDNN/trainset/nonfish/templ_HB70_LR_camB_Templ_651.jpg", cv::IMREAD_UNCHANGED );

  cv::Mat tFImage = cv::imread( "/home/kostasl/workspace/hello_tf_c_api/build/fish_sample2.jpg", cv::IMREAD_UNCHANGED );
  cv::Mat tFImageB = cv::imread( "/home/kostasl/workspace/zebrafishtrack/tensorDNN/trainset/fish/templ_HB150_NF0_6dpf_LR_camB_Templ_35569.jpg", cv::IMREAD_GRAYSCALE );

  std::cout<< "<<< Identify NON FISH >>> " <<std::endl;
  run_single_image_example(tNFImage);
  std::cout << std::endl;

  run_single_image_example(tNFImageB);
  std::cin.get();
  std::cout << std::endl;

  std::cout<< "<<< Identify FISH >>> " <<std::endl<<std::endl<<std::endl;
  run_single_image_example(tFImage);
  std::cout << std::endl;


  run_single_image_example(tFImageB);
  std::cin.get();
  std::cout << std::endl;

  //std::cout << "Running the INPUT_IMAGE -> GET_IMAGE example:" << std::endl;
  //run_expecting_image_example();
  //std::cin.get();

  return 0;
}

/// Example from https://github.com/kostasl/tensorflow_capi_sample
void runSavedModel(cv::Mat tImage)
{
    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;



    const char* saved_model_dir = "/home/kostasl/workspace/zebrafishtrack/tensorDNN/savedmodels/fishNet_prob"; // Path of the model
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    //Next we grab the tensor node from the graph by their name
    //****** Get input tensor
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);
    /// To obtain the names I used : saved_model_cli.py show --dir /home/kostasl/workspace/zebrafishtrack/tensorDNN/savedmodels/fishNet_prob/ --tag_set serve --signature_def serving_default
    //For Model no Softmax output layer the input layer was called "serving_default_sequential_input"
    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_sequential_1_input" ), 0}; //"serving_default_sequential_input"
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName sequential input //serving_default_sequential_1_input\n");
    else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

    Input[0] = t0;

    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
    printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    Output[0] = t2;


    //image.convertTo(image,CV_32FC1);
    cv::normalize(tImage,tImage,1.0,0,cv::NORM_MINMAX,CV_32FC1);
    cv::imshow("NORM Input img",tImage);

    cv::waitKey(1000); //For Img Rendering to finish

    std::cout << "Test TF Code" << std::endl;
    //********* Allocate data for inputs & outputs
        TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
        TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

        int ndims = 4;
        int64_t dims[] = {1,tImage.cols,tImage.rows,tImage.channels()};
        float* data = (float*)tImage.data;// {20};
        int ndata = sizeof(float)*tImage.total()*tImage.channels(); // This is tricky, it number of bytes not number of element
        std::cout << "~~Image size px*chan:"<< tImage.total()*tImage.channels() << " bytes: " << ndata << std::endl;

        TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
        if (int_tensor != NULL)
        {
            printf("TF_NewTensor is OK\n");
        }
        else
        printf("ERROR: Failed TF_NewTensor\n");

        InputValues[0] = int_tensor;


        printf("Attempting to run session...");//Inp:%f Ival:%f" //Input,InputValues
    // //Run the Session
        TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);

        if(TF_GetCode(Status) == TF_OK)
        {
            printf("Session is OK\n");
        }
        else
        {
            printf("%s",TF_Message(Status));
        }
        /// \brief Lastly, we want get back the output value from the output tensor using TF_TensorData that extract data from the tensor object.
        ///  Since we know the size of the output which is 1, i can directly print it. Else use TF_GraphGetTensorNumDims or other API that is available in c_api.h or tf_tensor.h
        int dims_out= TF_GraphGetTensorNumDims(Graph,t2,Status);
        if (TF_GetCode(Status) == TF_OK)
             printf("Get Out Dims ret: %d OK\n,",dims_out);
        else
             printf("%s",TF_Message(Status));

        void* buff = TF_TensorData(OutputValues[0]);
        float* offsets = (float*)buff;
        printf("Result Tensor :\n");
        printf("%f,%f\n",offsets[0],offsets[1]);



        // //Free memory
        TF_DeleteGraph(Graph);
        TF_DeleteSession(Session, Status);
        TF_DeleteSessionOptions(SessionOpts);
        TF_DeleteStatus(Status);


}


/// Run example code loading a non-frozen - SavedModel of fishNet that classifies 28x38 Greyscale images whether they contain a zebrafish head
void run_single_image_example(cv::Mat image)
{
  // Only 20% of the available GPU memory will be allocated
  float gpu_memory_fraction = 0.2f;

  // the model will try to infer the input and output layer names automatically 
  // (only use if it's a simple "one-input -> one-output" model
  bool inferInputOutput = false;

  // load a model from a .pb file
  tf_image::TF_Model model1;

  model1.loadModel("/home/kostasl/workspace/zebrafishtrack/tensorDNN/savedmodels/fishNet_prob/" , gpu_memory_fraction, inferInputOutput );//"graph_im2vec.pb"
  model1.setInputs( { "serving_default_sequential_1_input" } );
  model1.setOutputs( { "StatefulPartitionedCall" } );


  //std::cout << "* Load Image..." << std::endl;
  //cv::Mat image = cv::imread( "image.jpg", cv::IMREAD_UNCHANGED );
  // resize the image to fit the model's input:
  //cv::resize( image, image, { 244,244 } );

  // run prediction:  
  std::cout << "*  run prediction..." << std::endl;
  std::vector< std::vector< float > > results = model1.predict<std::vector<float>>( { image } );
  //   ^              ^ second vector is a normal model output (i.e. for classification or regression)
  //   ^ the elements of the first vector correspond to the model's outputs (if the model has only one, the vector contains only 1 vector)

  // print results
  std::cout << "* print results n:" << results.size() << std::endl;
  for ( size_t i = 0; i < results.size(); i++ )
  {
    std::cout << "Output vector #" << i << ": ";
    for ( size_t j = 0; j < results[i].size(); j++ )
    {
      std::cout << std::fixed << std::setprecision(4) << results[i][j] << "\t";
    }
    std::cout << std::endl;
  }
}

void run_expecting_image_example() {

  // Only 20% of the available GPU memory will be allocated
  float gpu_memory_fraction = 0.2f;

  // the model will try to infer the input and output layer names automatically 
  // (only use if it's a simple "one-input -> one-output" model
  bool inferInputOutput = true;

  // load a model from a .pb file
  tf_image::TF_Model model2;
  model2.loadModel( "graph_im2im.pb", gpu_memory_fraction, inferInputOutput );

  // load input image
  cv::Mat image = cv::imread( "46745.png", cv::IMREAD_UNCHANGED );
  
  // run prediction
  std::vector<cv::Mat> result = model2.predict<cv::Mat>( { image } );
  // the output image is type float32, and it can also contain any number of channels (even more than 4)

  // we can try to visualize it like a heatmap:
  cv::Mat heatmap = createHeatmap( result[0] );
  cv::resize( heatmap, heatmap, image.size() );

  std::cout << "Showing the image" << std::endl;
  while ( cv::waitKey( 1 ) != 27 ) {
    cv::imshow( "original input image", image );
    cv::imshow( "output heatmap", heatmap );
  }
  cv::destroyAllWindows();
}

cv::Mat createHeatmap( const cv::Mat& heatmaps ) {
  cv::Mat hue_ch = cv::Mat::zeros( heatmaps.rows, heatmaps.cols, CV_8U );
  cv::Mat sat_ch = cv::Mat::ones( heatmaps.rows, heatmaps.cols, CV_8U ) * 255;
  cv::Mat val_ch = cv::Mat::ones( heatmaps.rows, heatmaps.cols, CV_8U ) * 255;

  for ( int i = 0; i < heatmaps.channels(); i++ ) {
    cv::Mat h_ch, h_ch_uint8;
    cv::extractChannel( heatmaps, h_ch, i );

    h_ch *= 180;

    h_ch.convertTo( h_ch_uint8, CV_8U );

    hue_ch |= h_ch_uint8;
  }

  cv::Mat prettyHeatmap;
  cv::merge( std::vector<cv::Mat> { hue_ch, sat_ch, val_ch }, prettyHeatmap );

  cv::cvtColor( prettyHeatmap, prettyHeatmap, cv::COLOR_HSV2RGB );

  return prettyHeatmap;
}

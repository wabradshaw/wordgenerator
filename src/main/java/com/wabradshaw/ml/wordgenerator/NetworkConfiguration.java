package com.wabradshaw.ml.wordgenerator;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Data class containing the configuration settings for the machine learning model.
 */
public class NetworkConfiguration {
    private final int layerSize;
    private final TokenSet tokenSet;
    private final double learningRate;
    private final int seed;

    private final MultiLayerConfiguration config;

    /**
     * @param tokenSet The enum representing the string tokens that could be generated by the network
     * @param layerSize The number of short term memory nodes in each hidden layer of the network (i.e. not in/out)
     * @param learningRate How quickly the network should update parameters
     * @param seed The seed to use for randomness
     */
    public NetworkConfiguration(TokenSet tokenSet, int layerSize, double learningRate, int seed) {
        this.tokenSet = tokenSet;
        this.layerSize = layerSize;
        this.learningRate = learningRate;
        this.seed = seed;

        this.config = buildConfig();
    }

    /**
     * Creates and intialises a new Neural Network based on the settings in this config object.
     */
    public MultiLayerNetwork createNetwork(){
        MultiLayerNetwork network = new MultiLayerNetwork(this.config);
        network.init();
        return network;
    }

    /**
     * Builds the configuration object for Neural Networks using these settings.
     *
     * @return A {@link MultiLayerConfiguration} defining the Neural Network.
     */
    private MultiLayerConfiguration buildConfig() {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(learningRate))
                .list()
                .layer(0, new LSTM.Builder()
                                       .nIn(tokenSet.getLength())
                                       .nOut(layerSize)
                                       .activation(Activation.TANH)
                                       .build())
                .layer(1, new LSTM.Builder()
                                       .nIn(layerSize)
                                       .nOut(layerSize)
                                       .activation(Activation.TANH)
                                       .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                 .activation(Activation.SOFTMAX)
                                                 .nIn(layerSize)
                                                 .nOut(tokenSet.getLength())
                                                 .build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTLength(5)
                .pretrain(false).backprop(true)
                .build();
    }

}

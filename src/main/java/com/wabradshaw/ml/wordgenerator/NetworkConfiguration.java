package com.wabradshaw.ml.wordgenerator;

import com.wabradshaw.ml.wordgenerator.tokenisation.LexemeTokeniser;
import com.wabradshaw.ml.wordgenerator.tokenisation.PhonemeTokeniser;
import com.wabradshaw.ml.wordgenerator.tokenisation.Tokeniser;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Data class containing the configuration settings for the machine learning model.
 */
public class NetworkConfiguration {
    public static final String[] CHARS_EN_CAPS = "^$ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    public static final String[] CHARS_EN_CAPS_WITH_COMMON = "^$ABCDEFGHIJKLMNOPQRSTUVWXYZ-'".split("");

    public static final String[] PHONEMES_ARPABET_SEPARATE_STRESSES = ("^,AA,AE,AH,AO,AW,AY,B,CH,D,DH,EH,ER,EY,F,G,HH," +
            "IH,IY,JH,K,L,M,N,NG,OW,OY,P,R,S,SH,T,TH,UH,UW,V,W,Y,Z,ZH,0,1,2").split(",");

    public static final String[] PHONEMES_ARPABET_STRESSED = ("^,AA,AA0,AA1,AA2,AE,AE0,AE1,AE2,AH,AH0,AH1,AH2,AO," +
            "AO0,AO1,AO2,AW,AW0,AW1,AW2,AY,AY0,AY1,AY2,B,CH,D,DH,EH,EH0,EH1,EH2,ER,ER0,ER1,ER2,EY,EY0,EY1,EY2,F,G,HH," +
            "IH,IH0,IH1,IH2,IY,IY0,IY1,IY2,JH,K,L,M,N,NG,OW,OW0,OW1,OW2,OY,OY0,OY1,OY2,P,R,S,SH,T,TH,UH,UH0,UH1,UH2," +
            "UW,UW0,UW1,UW2,V,W,Y,Z,ZH").split(",");


    private final Tokeniser tokeniser;

    private final int layerSize;
    private final String[] tokenSet;
    private final double learningRate;
    private final int seed;

    private final MultiLayerConfiguration config;

    /**
     * @param mode Whether the system is generating lexemes or phonemes
     * @param tokenSet The list of string tokens that could be generated by the network
     * @param layerSize The number of short term memory nodes in each hidden layer of the network (i.e. not in/out)
     * @param learningRate How quickly the network should update parameters
     * @param seed The seed to use for randomness
     */
    public NetworkConfiguration(Mode mode, String[] tokenSet, int layerSize, double learningRate, int seed) {
        this.tokeniser = buildTokeniser(mode, tokenSet);

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
        network.setListeners(new ScoreIterationListener(1));
        return network;
    }

    /**
     * Builds the {@link Tokeniser} based on the {@link Mode} of the system.
     *
     * @return A {@link Tokeniser}
     */
    private Tokeniser buildTokeniser(Mode mode, String[] tokenSet){
        switch(mode){
            case LEXEMES:
                return new LexemeTokeniser(tokenSet);
            case PHONEMES:
                return new PhonemeTokeniser(tokenSet);
            default:
                throw new IllegalArgumentException("Please supply an operation Mode.");
        }
    }

    /**
     * Builds the configuration object for Neural Networks using these settings.
     *
     * @return A {@link MultiLayerConfiguration} defining the Neural Network.
     */
    private MultiLayerConfiguration buildConfig() {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(learningRate))
                .list()
                .layer(0, new LSTM.Builder()
                                       .nIn(tokenSet.length)
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
                                                 .nOut(tokenSet.length)
                                                 .build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTLength(30)
                .pretrain(false).backprop(true)
                .build();
    }

    public Tokeniser getTokeniser() {
        return this.tokeniser;
    }

}

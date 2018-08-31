package com.wabradshaw.ml.wordgenerator;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

/**
 * Data class containing the configuration settings for the machine learning model.
 */
public class NetworkConfiguration {
    public static final String[] CHARS_EN_CAPS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    public static final String[] CHARS_EN_CAPS_WITH_COMMON = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-'".split("");

    public static final String[] PHONEMES_ARPABET_SEPARATE_STRESSES = ("AA,AE,AH,AO,AW,AY,B,CH,D,DH,EH,ER,EY,F,G,HH," +
            "IH,IY,JH,K,L,M,N,NG,OW,OY,P,R,S,SH,T,TH,UH,UW,V,W,Y,Z,ZH,0,1,2").split(",");

    public static final String[] PHONEMES_ARPABET_STRESSED = ("AA,AA0,AA1,AA2,AE,AE0,AE1,AE2,AH,AH0,AH1,AH2,AO," +
            "AO0,AO1,AO2,AW,AW0,AW1,AW2,AY,AY0,AY1,AY2,B,CH,D,DH,EH,EH0,EH1,EH2,ER,ER0,ER1,ER2,EY,EY0,EY1,EY2,F,G,HH," +
            "IH,IH0,IH1,IH2,IY,IY0,IY1,IY2,JH,K,L,M,N,NG,OW,OW0,OW1,OW2,OY,OY0,OY1,OY2,P,R,S,SH,T,TH,UH,UH0,UH1,UH2," +
            "UW,UW0,UW1,UW2,V,W,Y,Z,ZH").split(",");

    private final int layerSize;
    private final int seed;

    /**
     * @param layerSize The number of short term memory nodes in each hidden layer of the network (i.e. not in/out)
     * @param seed The seed to use for randomness
     */
    public NetworkConfiguration(int layerSize, String[] charSet, int seed) {
        this.layerSize = layerSize;
        this.seed = seed;
    }

    public MultiLayerConfiguration build() {
        return null;
    }
}

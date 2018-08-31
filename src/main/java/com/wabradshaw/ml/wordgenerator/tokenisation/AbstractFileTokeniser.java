package com.wabradshaw.ml.wordgenerator.tokenisation;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class AbstractFileTokeniser implements Tokeniser {

    protected final String[] possibleTokens;

    public AbstractFileTokeniser(String[] possibleTokens){
        this.possibleTokens = possibleTokens;
    }

    @Override
    public INDArray getTokens() {
        // Load file
        // Get lines
        // Remove bad lines

        // Split according to implementation
        // Tokenise according to implementation

        // Vectorise
        // Convert to INDArray
        return null;
    }
}

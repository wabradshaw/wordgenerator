package com.wabradshaw.ml.wordgenerator.tokenisation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Interface for classes which convert portions of the word-phoneme data into vector tokens.
 */
public interface Tokeniser {

    /**
     * Main point of tokenisers. Reads in training data and converts each line into an array of character-vectors.
     *
     * @return
     */
    public DataSet getTokens();

    public String getToken(int index);
}

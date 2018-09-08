package com.wabradshaw.ml.wordgenerator.tokenisation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

/**
 * Interface for classes which convert portions of the word-phoneme data into vector tokens.
 */
public interface Tokeniser {

    public String getRelevantWord(String line);

    public List<Integer> tokenise(String word);

    public int toToken(String symbol);

    public String toSymbol(int index);
}

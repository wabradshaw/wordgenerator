package com.wabradshaw.ml.wordgenerator.tokenisation;

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

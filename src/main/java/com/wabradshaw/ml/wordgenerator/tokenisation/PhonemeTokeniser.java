package com.wabradshaw.ml.wordgenerator.tokenisation;

import java.util.List;

public class PhonemeTokeniser extends AbstractFileTokeniser {

    public PhonemeTokeniser(String[] possibleTokens) {
        super(possibleTokens);
    }

    @Override
    protected List<Integer> toTokens(String word) {
        return null;
    }

    @Override
    protected String getRelevantWord(String line) {
        return null;
    }
}

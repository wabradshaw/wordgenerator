package com.wabradshaw.ml.wordgenerator.tokenisation;

import com.wabradshaw.ml.wordgenerator.TokenSet;

import java.util.List;

public class PhonemeTokeniser extends AbstractTokeniser {

    public PhonemeTokeniser(TokenSet tokenSet) {
        super(tokenSet);
    }

    @Override
    public List<Integer> tokenise(String word) {
        return null;
    }

    @Override
    public String getRelevantWord(String line) {
        return null;
    }
}

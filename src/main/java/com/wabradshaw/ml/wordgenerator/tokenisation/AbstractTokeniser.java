package com.wabradshaw.ml.wordgenerator.tokenisation;

import com.wabradshaw.ml.wordgenerator.TokenSet;

import java.util.HashMap;
import java.util.Map;

public abstract class AbstractTokeniser implements Tokeniser{

    private final String[] tokens;
    private final Map<String, Integer> tokenToIndexMap;

    public AbstractTokeniser(TokenSet tokenSet){
        this.tokens = tokenSet.getTokens();
        this.tokenToIndexMap = new HashMap<>();

        for(int i = 0; i < tokenSet.getLength(); i++){
            tokenToIndexMap.put(tokens[i], i);
        }
    }

    @Override
    public int toToken(String symbol){
        return tokenToIndexMap.get(symbol);
    }

    @Override
    public String toSymbol(int index){
        return tokens[index];
    }

    protected boolean knownSymbol(String symbol){
        return tokenToIndexMap.containsKey(symbol);
    }
}

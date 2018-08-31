package com.wabradshaw.ml.wordgenerator;

/**
 * The mode describing what type of generation is being performed. E.g. Is it generating spelling or pronunciation.
 */
public enum Mode {

    /**
     * The system is generating the spelling for a new word.
     */
    LEXEMES,

    /**
     * The system is generating the pronunciation for a new word.
     */
    PHONEMES;
}

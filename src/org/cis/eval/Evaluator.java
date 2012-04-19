package org.cis.eval;

import org.cis.cf.algorithm.RatingPredictor;
import org.cis.data.Ratings;

public interface Evaluator {
	
	public double evaluate(RatingPredictor rp, Ratings ratings);
	
}

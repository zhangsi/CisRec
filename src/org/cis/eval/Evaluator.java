package org.cis.eval;

import org.cis.cf.algorithm.RatingPredictor;
import org.cis.data.Ratings;

/**
 * This interface defining evaluation function
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public interface Evaluator {
	
	/**
	 * Evaluate the performance the RatingPredictor on validation Ratings data 
	 * @param rp the RatingPredictor to be evaluate
	 * @param ratings The validation data of ratings
	 * @return the performance
	 */
	public double evaluate(RatingPredictor rp, Ratings ratings);
	
}

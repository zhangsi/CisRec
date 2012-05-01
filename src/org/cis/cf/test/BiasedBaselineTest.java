package org.cis.cf.test;

import org.cis.cf.algorithm.BiasedBaseline;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.MovielensRatingsReader;

/**
 * This class tests the BiasedBaseline algorithm
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class BiasedBaselineTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.BiasedBaselineTest ../data/movielens/u1.base ../data/movielens/u1.test 0.05 50
		
		
		if(args.length != 4){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
			System.out.println("3, Learning rate");
			System.out.println("4, Max round of training");
		} else {
			String trainFile = args[0];
			String testFile = args[1];
			
			double learnRate = Double.parseDouble(args[2]);
			int maxIter = Integer.parseInt(args[3]);
			
			MovielensRatingsReader reader = new MovielensRatingsReader();
			Ratings trainData  = reader.read(trainFile);
			Ratings testData   = reader.read(testFile);
			
			BiasedBaseline recommender = new BiasedBaseline(
					trainData,
					maxIter,
					learnRate
					);
			
			recommender.trainModel();
			
			RmseEvaluator evaluator = new RmseEvaluator();
			System.out.println(evaluator.evaluate(recommender, testData));
		}
	}
}

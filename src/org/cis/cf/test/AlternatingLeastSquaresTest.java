package org.cis.cf.test;

import org.cis.cf.algorithm.AlternatingLeastSquares;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.MovielensRatingsReader;

public class AlternatingLeastSquaresTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.AlternatingLeastSquaresTest ../data/movielens/u1.base ../data/movielens/u1.test 10 0.125 0.125 25
		
		
		if(args.length != 6){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
			System.out.println("3, Number of latent factors");
			System.out.println("4, Regularization coeffient for user factors");
			System.out.println("5, Regularization coeffient for item factors");
			System.out.println("6, Max round of training");
		} else {
			String trainFile = args[0];
			String testFile = args[1];

			int featureNumber = Integer.parseInt(args[2]);
			double userReg = Double.parseDouble(args[3]);
			double itemReg = Double.parseDouble(args[4]);
			int maxIter = Integer.parseInt(args[5]);
			
			MovielensRatingsReader read = new MovielensRatingsReader();
			Ratings trainData  = read.read(trainFile);
			Ratings testData   = read.read(testFile);
			
			AlternatingLeastSquares recommender = new AlternatingLeastSquares(
					trainData,
					featureNumber,
					userReg,
					itemReg,
					maxIter
					);
			
			recommender.trainModel();
			
			RmseEvaluator evaluator = new RmseEvaluator();
			System.out.println(evaluator.evaluate(recommender, testData));
		}
	}
}

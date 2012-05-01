package org.cis.cf.test;

import org.cis.cf.algorithm.ProbabilisticMatrixFactorization;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.MovielensRatingsReader;

/**
 * This class tests the Probabilistic Matrix Factorization algorithm
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class ProbabilisticMatrixFactorizationTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.ProbabilisticMatrixFactorizationTest ../data/movielens/u1.base ../data/movielens/u1.test 10 0.01 0.01 0.01 25
		
		
		if(args.length != 7){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
			System.out.println("3, Number of latent factors");
			System.out.println("4, Learning rate");
			System.out.println("5, Regularization coeffient for user factors");
			System.out.println("6, Regularization coeffient for item factors");
			System.out.println("7, Max round of training");
		} else {
			String trainFile = args[0];
			String testFile = args[1];

			int featureNumber = Integer.parseInt(args[2]);
			double learnRate = Double.parseDouble(args[3]);
			double userReg = Double.parseDouble(args[4]);
			double itemReg = Double.parseDouble(args[5]);
			int maxIter = Integer.parseInt(args[6]);
			
			MovielensRatingsReader read = new MovielensRatingsReader();
			Ratings trainData  = read.read(trainFile);
			Ratings testData   = read.read(testFile);
			
			ProbabilisticMatrixFactorization recommender = new ProbabilisticMatrixFactorization(
					trainData,
					featureNumber,
					learnRate,
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

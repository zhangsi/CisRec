package org.cis.cf.test;

import org.cis.cf.algorithm.BiasedProbabilisticMatrixFactorization;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.MovielensRatingsReader;

/**
 * This class tests the Biased Probabilistic Matrix Factorization algorithm
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class BiasedProbabilisticMatrixFactorizationTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.BiasedProbabilisticMatrixFactorizationTest ../data/movielens/u1.base ../data/movielens/u1.test 10 0.01 0.01 0.01 1 0.01 0.01 25
		
		
		if(args.length != 10){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
			System.out.println("3, Number of latent factors");
			System.out.println("4, Learning rate");
			System.out.println("5, Regularization coeffient for user factors");
			System.out.println("6, Regularization coeffient for item factors");
			System.out.println("7, Biased learning rate");
			System.out.println("8, Regularization coeffient for user bias");
			System.out.println("9, Regularization coeffient for item bias");
			System.out.println("10, Max round of training");
		} else {
			String trainFile = args[0];
			String testFile = args[1];

			int featureNumber = Integer.parseInt(args[2]);
			
			double learnRate = Double.parseDouble(args[3]);
			double userReg   = Double.parseDouble(args[4]);
			double itemReg   = Double.parseDouble(args[5]);
			
			double biasLearnRate = Double.parseDouble(args[6]);
			double biasUserReg   = Double.parseDouble(args[7]);
			double biasItemReg   = Double.parseDouble(args[8]);
			
			int maxIter = Integer.parseInt(args[9]);
			
			MovielensRatingsReader read = new MovielensRatingsReader();
			Ratings trainData  = read.read(trainFile);
			Ratings testData   = read.read(testFile);
			
			BiasedProbabilisticMatrixFactorization recommender = new BiasedProbabilisticMatrixFactorization(
					trainData,
					featureNumber,
					learnRate,
					userReg,
					itemReg,
					biasLearnRate,
					biasUserReg,
					biasItemReg,
					maxIter
					);
			
			recommender.trainModel();
			
			RmseEvaluator evaluator = new RmseEvaluator();
			System.out.println(evaluator.evaluate(recommender, testData));
		}
	}
}

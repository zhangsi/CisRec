package org.cis.cf.test;

import org.cis.cf.algorithm.ProbabilisticLatentSemanticAnalysis;
import org.cis.data.Ratings;
import org.cis.eval.RmseEvaluator;
import org.cis.io.MovielensRatingsReader;

/**
 * This class tests the Probabilistic Latent Semantic Analysis algorithm
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class ProbabilisticLatentSemanticAnalysisTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.ProbabilisticLatentSemanticAnalysisTest ../data/movielens/u1.base ../data/movielens/u1.test 10 5 1 20
		
		
		if(args.length != 6){
			System.out.println("Useage:");
			System.out.println("1, Training data path");
			System.out.println("2, Test data path");
			System.out.println("3, Number of latent factors");
			System.out.println("4, rating");
			System.out.println("5, beta");
			System.out.println("6, Max round of training");
		} else {
			String trainFile = args[0];
			String testFile = args[1];

			int featureNumber = Integer.parseInt(args[2]);
			
			int rating = Integer.parseInt(args[3]);
			double beta = Double.parseDouble(args[4]);
			int maxIter = Integer.parseInt(args[5]);
			
			MovielensRatingsReader read = new MovielensRatingsReader();
			Ratings trainData  = read.read(trainFile);
			Ratings testData   = read.read(testFile);
			
			ProbabilisticLatentSemanticAnalysis recommender = new ProbabilisticLatentSemanticAnalysis(
					trainData,
					featureNumber,
					rating,
					beta,
					maxIter
					);
			
			recommender.trainModel();
			
			RmseEvaluator evaluator = new RmseEvaluator();
			System.out.println(evaluator.evaluate(recommender, testData));
		}
	}
}

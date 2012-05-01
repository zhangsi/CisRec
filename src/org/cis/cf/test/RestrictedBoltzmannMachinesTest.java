package org.cis.cf.test;

import org.cis.cf.algorithm.RestrictedBoltzmannMachines;
import org.cis.data.Ratings;
import org.cis.io.NetflixRatingsReader;

/**
 * This class tests the Restricted Boltzmann Machines algorithm
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class RestrictedBoltzmannMachinesTest {
	public static void main(String args[]){
		
		
		//command: java org.cis.cf.test.RestrictedBoltzmannMachinesTest ../data/netflix/train ../data/netflix/probe 20 5 0.01 0.08 0.006 0.01 0.8 0.9 300
		
		if(args.length != 11){
			System.out.println("Useage:");
			System.out.println("1,  Training data path");
			System.out.println("2,  Test data path");
			System.out.println("3,  Number of latent factors");
			System.out.println("4,  softmax");
			System.out.println("5,  Learning rate for weights");
			System.out.println("6,  Learning rate for biases of visible units");
			System.out.println("7,  Learning rate for biases of hidden units");
			System.out.println("8,  weight cost");
			System.out.println("9,  momentum");
			System.out.println("10, fianl momentum");
			System.out.println("11, Max round of training");
		} else {
			String trainFile = args[0];
			String testFile = args[1];

			int featureNumber = Integer.parseInt(args[2]);
			int softmax       = Integer.parseInt(args[3]);
			double epsilonw   = Double.parseDouble(args[4]);
			double epsilonvb  = Double.parseDouble(args[5]);
			double epsilonhb  = Double.parseDouble(args[6]);
			double weightCost = Double.parseDouble(args[7]);
			double momentum   = Double.parseDouble(args[8]);
			double finalMomentum = Double.parseDouble(args[9]);
			
			int maxIter = Integer.parseInt(args[10]);

			
			NetflixRatingsReader reader = new NetflixRatingsReader();
			Ratings trainData  = reader.read(trainFile);
			Ratings testData   = reader.read(testFile);
			
			RestrictedBoltzmannMachines recommender = new RestrictedBoltzmannMachines(
					trainData,
					testData,
					featureNumber,
					softmax,
					maxIter,
					epsilonw,
					epsilonvb,
					epsilonhb,
					weightCost,
					momentum,
					finalMomentum
					);
			
			recommender.trainModel();

		}
	}
}

import java.io.File;
import java.io.IOException;
import java.util.*;

public class LogisticLearner {
	
	Double eta = 0.0; //Tunable parameter for gradient descent
	ClassInfo[] classes;
	HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>(); //occurrences of each class
	ArrayList<int[]> trainingData = new ArrayList<int[]>(); //training data as usable arraylist
	ArrayList<int[]> testData = new ArrayList<int[]>(); //test data as usable arraylist
	ArrayList<int[]> validationSet = new ArrayList<int[]>(); //validation set as usable arraylist
	String dataSetName; //the name of the dataset being tested/trained
	
	public LogisticLearner(String name){
		this.dataSetName = name;
	}
	
	/************************************************************
	Tester method.
	************************************************************/
	
	public static void main(String[] args) throws IOException {
		String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
		for(int j = 0; j < 5; j++){
			NaiveBayes b = new NaiveBayes(dataSets[j]);
			System.out.println(dataSets[j] + ":");
			System.out.println();
			System.out.println("Logisitic Regression");
			for(int i = 1; i < 6; i++){
				double bestEta =.001;
				LogisticLearner loglearn = new LogisticLearner(dataSets[j]);
				loglearn.eta += bestEta;
				loglearn.fillTrainFile(i);
				loglearn.fillTestFile(i);
				loglearn.fillValidationSet();
				loglearn.countClasses();
				loglearn.initializeClasses();
				loglearn.gradientDescent();
				System.out.printf("%.2f",loglearn.test()*100);
				System.out.println();
				
			}
			System.out.println();
			System.out.println("Naive Bayes:");
			for(int i = 1; i < 6; i++){
				b.fillTrainFile(i);
				b.fillTestFile(i);
				b.countClasses();
				for(int key : b.classCounts.keySet()){
					b.trainNaiveBayes(b.trainingData, key);
				}
				b.testNaiveBayes(b.testData);
			}
			System.out.println("------------------------------");
		}
	}

	
	
	/************************************************************
	Initialize Weights for Gradient Descent.
	************************************************************/
	
	/************************************************************
	Method to test logistic learner. Uses test data.
	************************************************************/
	
	public double test(){
		double performance = 0;
		for(int[] arr : testData){
			double max = Double.MIN_VALUE;
			Integer classification = -1;
			for(ClassInfo c : classes){
				c.resetOutput(); // set o = 0
			}
			for(ClassInfo c: classes){
				c.dotProduct(arr);
			}
			for(ClassInfo c : classes){
				c.probability = Math.exp(c.output)/getDenominator(); //calculate class probability.
				if(c.probability > max){
					max = c.probability;
					classification = c.classVal; //set classification.
				}
			}
			if(classification == arr[arr.length-1]){
				performance++;//incrememnt performance if class values match.
			}
		}
		return (performance / ((double) testData.size()));
	}
	
	/************************************************************
	This is the gradient descent method for learning the weights
	for logisic regression.
	************************************************************/
	
	public double gradientDescent(){
		
		initializeWeights(); //initialize weights of classes. (see below)
		ClassInfo[] classesCopy = new ClassInfo[classes.length]; //initialize copy of classes
		double performance = Double.MIN_VALUE;
		double currentPerformance = verifyPerformance(); // set initial performance value based on random class weights
		while(currentPerformance >= performance){ //stopping condition.
			performance = currentPerformance; //update performance
			//reset deltas of each class
			for(ClassInfo c: classes){
				c.resetDeltas(); // reset the delta_weights to an initial value.
			}
			
			//MAIN LOOP
			
			for(int[] arr : trainingData){ // iterate through each training example
				
				for(ClassInfo c : classes){
					c.resetOutput(); // set "o" = 0
				}
				
				//set o for each class
				for(ClassInfo c : classes){
					c.dotProduct(arr); // sets "o" for each class
				}
				
				//set probability for each class
				for(ClassInfo c: classes){
					c.probability = Math.exp(c.output + c.w0)/(getDenominator()); // probability of each class given the training example
				}
				
				//calculate and set deltas
				for(ClassInfo c : classes){
					if(c.classVal == arr[arr.length-1]){ // implementation of konecker delta.
						for(int i = 0; i < arr.length-1; i++){
							c.deltaWeights[i] = c.deltaWeights[i] + (1 - c.probability)*((double)arr[i]);
						}
						c.deltaW0 = c.deltaW0 + (1-c.probability);
					}else{
						for(int i = 0; i < arr.length-1; i++){
							c.deltaWeights[i] = c.deltaWeights[i] + (0 - c.probability)*((double)arr[i]); //penalize if classes dont match
						}
						c.deltaW0 = c.deltaW0 + (0-c.probability);
					}
				}
			}
			//set weights
			classesCopy = classes; // save class weights from previous iteration before changing them.
			for(ClassInfo c : classes){
				for(int i = 0; i < c.weights.length; i++){
					c.weights[i] = c.weights[i] + eta*c.deltaWeights[i]; // update weights. 
				}
				c.w0 = c.w0 + eta*c.deltaW0;
			}
			
			//test performance
			currentPerformance = verifyPerformance(); // calculate new performance based on validation set.
			if(currentPerformance > .8){ //if the performance is above 80% break.
				break;
			}else if(currentPerformance < performance){ //otherwise use the old class weights as the weights.
				classes = classesCopy;
				break;
			}
		}
		return currentPerformance;
	}
	
	/************************************************************
	This method is used to help gradient descent stop early. This
	calculates the performance of the weight set when a validation
	set is applied. Identical to the test method above.
	************************************************************/
	
	double verifyPerformance(){
		double performance = 0;
		for(int[] arr : validationSet){
			double max = Double.MIN_VALUE;
			Integer classification = -1;
			for(ClassInfo c : classes){
				c.resetOutput(); // set o = 0
			}
			for(ClassInfo c: classes){
				c.dotProduct(arr);
			}
			for(ClassInfo c : classes){
				c.probability = Math.exp(c.output)/getDenominator();
				if(c.probability > max){
					max = c.probability;
					classification = c.classVal;
				}
			}
			
			if(classification == arr[arr.length-1]){
				performance++;
			}
		}
		return (performance / ((double) validationSet.size()));
	}
	
	/************************************************************
	Initialize classes to be the class value.
	************************************************************/
	
	public void initializeClasses(){
		classes = new ClassInfo[classCounts.size()];
		int i = 0;
		for(Integer key: classCounts.keySet()){
			ClassInfo temp = new ClassInfo(key);
			classes[i] = temp;
			i++;
		}
	}
	
	/************************************************************
	Initializes the weights of each class. See ClassInfo.setWeights().
	************************************************************/
	
	void initializeWeights(){
		for(int i = 0; i < classes.length; i++){
			classes[i].weights = new double[trainingData.get(0).length-1];
			classes[i].deltaWeights = new double[trainingData.get(0).length-1];
			classes[i].setWeights();
		}
	}
	
	/************************************************************
	This method helps with calculating the probability of a class
	given a data point. This calculates the denomenator.
	************************************************************/
	
	double getDenominator(){
		double sum = 0;
		for(ClassInfo c : classes){
			sum += Math.exp(c.output + c.w0);
		}
		return sum;
	}
	
	
	/************************************************************
	
	Helper Methods
	
	************************************************************/
	
	/************************************************************
	Counts all of the classes in the training set. 
	************************************************************/
	
	public void countClasses(){
		//handle special cases.
		for(int i = 0; i < trainingData.size(); i++){
			if(classCounts.containsKey(trainingData.get(i)[trainingData.get(i).length-1])){ // hashmap contains class?
				int currentVal = classCounts.get(trainingData.get(i)[trainingData.get(i).length-1]); // get current count
				currentVal++; // increment count by 1
				classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], currentVal); //update map.
			}else{
				classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], 1); // add class to map
			}
		}
	}
	
	/************************************************************
	Fills the training file based on the index that is to be skipped.
	Used specifically for cross validation.
	************************************************************/
	
	void fillTrainFile(int indexToSkip) throws IOException{
		for(int i = 1; i < 6; i++){
			if(i == indexToSkip) continue; // skip file at index.
			Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+i+".txt"));
			while(fileScanner.hasNextLine()){
				String[] arr = fileScanner.nextLine().split(" ");

				int[] vals = new int[arr.length];
				for(int j = 0; j < arr.length; j++){
					vals[j] = Integer.parseInt(arr[j]);
				}
				trainingData.add(vals);
			}
			fileScanner.close();
		}
	}
	
	/************************************************************
	Fills test file similarly to fillTrainFile.
	************************************************************/
	
	void fillTestFile(int indexToSkip) throws IOException{
		Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+indexToSkip+".txt"));
		while(fileScanner.hasNextLine()){
			String[] arr = fileScanner.nextLine().split(" ");
			int[] vals = new int[arr.length];
			for(int j = 0; j < arr.length; j++){
				vals[j] = Integer.parseInt(arr[j]);
			}
			testData.add(vals);
		}
		fileScanner.close();
	}
	
	/************************************************************
	Fills the validation set.
	************************************************************/
	
	void fillValidationSet() throws IOException{
		Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/validationSet.txt"));
		while(fileScanner.hasNextLine()){
			String[] arr = fileScanner.nextLine().split(" ");

			int[] vals = new int[arr.length];
			for(int j = 0; j < arr.length; j++){
				vals[j] = Integer.parseInt(arr[j]);
			}
			validationSet.add(vals);
		}
		fileScanner.close();
	}
	
	/************************************************************
	
	Helper Class.
	
	This class allows for easy iteration and operations on all 
	of the classes.
	
	************************************************************/
	
	static class ClassInfo {
		Integer classVal = 0;
		Double probability = 0.0;
		double[] weights;
		double w0;
		double deltaW0;
		double[] deltaWeights;
		double output = 0;
		
		public ClassInfo(Integer val){
			this.classVal = val;
			w0 = .5; // tunable
			deltaW0 = .5; //tunable
		}
		
		/************************************************************
		Dots an input vector with the weights. Returns the absolute 
		value.
		************************************************************/
		
		void dotProduct(int[] arr){
			for(int i = 0; i < weights.length; i++){
				output += weights[i]*((double)arr[i]);
			}
			output = Math.abs(output);
		}
		
		/************************************************************
		Resets output to 0.
		************************************************************/
		
		void resetOutput(){
			output = 0;
		}
		
		/************************************************************
		Resets deltas to 0 and returns constants to initial values.
		************************************************************/
		
		void resetDeltas(){
			for(int i = 0; i < deltaWeights.length; i++){
				deltaWeights[i] = 0;
			}
			deltaW0 = .5;
			w0 = .5;
		}
		
		/************************************************************
		Sets random weights. The performance of the algorithm depends
		heavily on the distribution of the weights.
		************************************************************/
		
		void setWeights(){
			for(int i = 0; i < weights.length; i++){
				weights[i] = (0 + Math.random()*0.1);
			}
		}
	}
}

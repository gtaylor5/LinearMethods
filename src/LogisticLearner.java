import java.io.File;
import java.io.IOException;
import java.util.*;

public class LogisticLearner {
	
	Double eta = 0.0; //Tunable parameter for gradient descent
	ClassInfo[] classes;
	HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
	ArrayList<int[]> trainingData = new ArrayList<int[]>();
	ArrayList<int[]> testData = new ArrayList<int[]>();
	ArrayList<int[]> validationSet = new ArrayList<int[]>();
	String dataSetName;
	Integer dataDimension = 0;
	
	public LogisticLearner(String name){
		this.dataSetName = name;
	}
	
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
	NEED TO CALCULATE PROBABILITIES FOR K-1 CLASSES. THE KTH
	CLASS IS THE SUM OF ALL THE PROBABILITIES
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
				c.probability = Math.exp(c.output);
				if(c.probability > max){
					max = c.probability;
					classification = c.classVal;
				}
			}
			if(classification == arr[arr.length-1]){
				performance++;
			}
		}
		return (performance / ((double) testData.size()));
	}
	
	
	public double gradientDescent(){
		
		initializeWeights();
		ClassInfo[] classesCopy = new ClassInfo[classes.length];
		double performance = Double.MIN_VALUE;
		double currentPerformance = verifyPerformance();
		while(currentPerformance >= performance){
			performance = currentPerformance;
			//reset deltas of each class
			for(ClassInfo c: classes){
				c.resetDeltas();
			}
			
			//MAIN LOOP
			
			for(int[] arr : trainingData){
				
				for(ClassInfo c : classes){
					c.resetOutput(); // set o = 0
				}
				
				//set o for each class
				for(ClassInfo c : classes){
					c.dotProduct(arr); // sets o for each class
				}
				
				//set probability for each class
				for(ClassInfo c: classes){
					c.probability = Math.exp(c.output + c.w0)/(getDenominator());
				}
				
				//calculate and set deltas
				for(ClassInfo c : classes){
					if(c.classVal == arr[arr.length-1]){
						for(int i = 0; i < arr.length-1; i++){
							c.deltaWeights[i] = c.deltaWeights[i] + (1 - c.probability)*((double)arr[i]);
						}
						c.deltaW0 = c.deltaW0 + (1-c.probability);
					}else{
						for(int i = 0; i < arr.length-1; i++){
							c.deltaWeights[i] = c.deltaWeights[i] + (0 - c.probability)*((double)arr[i]);
						}
						c.deltaW0 = c.deltaW0 + (0-c.probability);
					}
				}
			}
			//set weights
			classesCopy = classes;
			for(ClassInfo c : classes){
				for(int i = 0; i < c.weights.length; i++){
					c.weights[i] = c.weights[i] + eta*c.deltaWeights[i];
				}
				c.w0 = c.w0 + eta*c.deltaW0;
			}
			
			//test performance
			currentPerformance = verifyPerformance();
			if(currentPerformance > .8){
				break;
			}else if(currentPerformance < performance){
				classes = classesCopy;
				break;
			}
		}
		return currentPerformance;
		
	}
	
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
	
	
	public void initializeClasses(){
		classes = new ClassInfo[classCounts.size()];
		int i = 0;
		for(Integer key: classCounts.keySet()){
			ClassInfo temp = new ClassInfo(key);
			classes[i] = temp;
			i++;
		}
	}
	
	void initializeWeights(){
		for(int i = 0; i < classes.length; i++){
			classes[i].weights = new double[trainingData.get(0).length-1];
			classes[i].deltaWeights = new double[trainingData.get(0).length-1];
			classes[i].setWeights();
		}
	}
	
	double getDenominator(){
		double sum = 0;
		for(ClassInfo c : classes){
			sum += Math.exp(c.output + c.w0);
		}
		return sum;
	}
	
	Double sigmoid(Double output, Integer classNumber){
		double value = 1;
		value /= (1+Math.exp((-1.0)*(output)));
		return value;
	}
	
	
	
	/************************************************************
	
	Helper Methods
	
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
	
	void fillTrainFile(int indexToSkip) throws IOException{
		for(int i = 1; i < 6; i++){
			if(i == indexToSkip) continue;
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
	
	void printArrayList(ArrayList<Double> list){
		for(Double val: list){
			System.out.print(val + " ");
		}
		System.out.println();
	}
	
	/************************************************************
	
	Helper Class
	
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
			w0 = .5;
			deltaW0 = .5;
		}
		
		void dotProduct(int[] arr){
			for(int i = 0; i < weights.length; i++){
				output += weights[i]*((double)arr[i]);
			}
			output = Math.abs(output);
		}
		
		void resetOutput(){
			output = 0;
		}
		
		void resetDeltas(){
			for(int i = 0; i < deltaWeights.length; i++){
				deltaWeights[i] = 0;
			}
			deltaW0 = .5;
			w0 = .5;
		}
		
		void setWeights(){
			for(int i = 0; i < weights.length; i++){
				weights[i] = (0 + Math.random()*0.1);
			}
		}
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}

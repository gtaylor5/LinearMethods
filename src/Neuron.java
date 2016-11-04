import java.util.*;


public class Neuron {
    
    Integer classVal;
    double theta = .02;
    double[] weights;
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    ArrayList<int[]> validationSet = new ArrayList<int[]>();
    
    /************************************************************
    Constructor
    ************************************************************/
    
    public Neuron(Integer classVal, ArrayList<int[]> train, ArrayList<int[]> valid){
        this.classVal = classVal;
        this.trainingData = train;
        this.validationSet = valid;
    }
    
    /************************************************************
    Test based on an input data point
    ************************************************************/
    
    int test(int[] arr){
        double val = sgn(weights, arr);
        if(val > 0 && classVal == arr[arr.length-1]){ // True positive only
            return 1;
        }
        return 0;
    }
    
    /************************************************************
    Train neuron based on class value.
    ************************************************************/
    
    public void train(){
        initializeWeights(); // intialize weights to random number vector
        double[] weightsCopy = weights; // save weights.
        double performance = Double.MIN_VALUE;
        double currentPerformance = verifyPerformance(); //get first performance
        while(currentPerformance > performance){ // stopping case
            weightsCopy = weights; // save previous weights prior to updating
            performance = currentPerformance; // update performance value
            for(int[] arr : trainingData){ //iterate through each data point in training set
                double output = sgn(weights, arr); // get the output of the sgn function given the weights and input point.
                if(output <= 0 && classVal == arr[arr.length-1]){ //False negative
                    for(int i = 0; i < arr.length-1; i++){
                        weights[i] += ((double)arr[i]);
                    }
                }else if(output >= 0 && classVal != arr[arr.length-1]){ //False Positive
                    for(int i = 0; i < arr.length-1; i++){
                        weights[i] -= ((double) arr[i]);
                    }
                }
            }
            currentPerformance = verifyPerformance(); // check performance against validation set
            if(currentPerformance >= .8){ // stopping case.
                break;
            }
        }
        weights = weightsCopy; // set weights to saved weights.
    }
    
    /************************************************************
    SigNum function.
    ************************************************************/
    
    int sgn(double[] weights, int[] arr){
        double val = dot(weights, arr);
        val += theta;
        if(val > 0){
            return 1;
        }
        return -1;
    }
    
    /************************************************************
    dot product of two arrays.
    ************************************************************/
    
    double dot(double[] weights, int[] arr) {
        double sum = 0;
        for(int i = 0; i < weights.length; i++){
            sum += weights[i]*((double)arr[i]);
        }
        return sum;
    }

    /************************************************************
    tests performance of learned weights against a validation set.
    Used in training process.
    ************************************************************/
    
    double verifyPerformance(){
        double performance = 0;
        for(int[] arr : validationSet){
            double val = sgn(weights, arr);
            if(val > 0 && classVal == arr[arr.length-1]){ // True Positive
                performance++;
            }else if(val <= 0 && classVal != arr[arr.length-1]){ //True Negative
                performance++;
            }
        }
        return (performance / ((double) validationSet.size()));
    }
    
    /************************************************************
    Initialize weights to value between 0 and 1.
    ************************************************************/
    
    public void initializeWeights(){
        weights = new double[trainingData.get(0).length-1];
        for(int i = 0; i < weights.length; i++){
            weights[i] = Math.random();
        }
    }
    
}

import java.util.ArrayList;
public class AdalineNeuron {
    Integer classVal;
    double eta = .5;
    double[] weights;
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    ArrayList<int[]> validationSet = new ArrayList<int[]>();
    
    /************************************************************
    Constructor
    ************************************************************/
    
    public AdalineNeuron(Integer classVal, ArrayList<int[]> train, ArrayList<int[]> valid){
        this.classVal = classVal;
        this.trainingData = train;
        this.validationSet = valid;
    }
    
    /************************************************************
    Test based on an input data point
    ************************************************************/
    
    int test(int[] arr){
        double val = dot(weights, arr);
        //System.out.println(val + " " + classVal + " " + arr[arr.length-1]);
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
        while(true){
            for(int[] arr : trainingData){
               double y = sgn(weights, arr);
                if(arr[arr.length-1] == classVal){
                    double target = 1;
                    double error = (target - y);
                    for(int i = 0; i < weights.length; i++){
                        weights[i] += eta*error*((double)arr[i]);
                    }
                }else{
                    double target = -1;
                    double error = (target - y);
                    for(int i = 0; i < weights.length; i++){
                        weights[i] += eta*error*((double)arr[i]);
                    }
                }
            }
            if(verifyPerformance() >= .8){
                break;
            }
        }
    }
    
    double getSumOfErrors(double[] err){
        double sum = 0;
        for(double val: err){
            sum+=val;
        }
        return sum;
    }
    
    /************************************************************
    SigNum function.
    ************************************************************/
    
    int sgn(double[] weights, int[] arr){
        double val = dot(weights, arr);
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
    
    double dotScalar(double err, int[] arr) {
        double sum = 0;
        for(int i = 0; i < weights.length; i++){
            sum += err*((double)arr[i]);
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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    ANN ann; // our artificial neural network
    double sumSquareError = 0; // the sum of the error squared

    private void Start()
    {
        ann = new ANN(2, 1, 1, 2, 0.8);

        List<double> result;

        // run 1000 epochs of the training set (XOR)
        // the sum of squared error should get smaller
        for (int i = 0; i < 1000; i++)
        {
            sumSquareError = 0;
            result = Train(1, 1, 0);
            // add the power of our result - expected result, to the power of 2
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);

            result = Train(1, 0, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);

            result = Train(0, 1, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);

            result = Train(0, 0, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
        }
        Debug.Log("SSE: " + sumSquareError);

        result = Train(1, 1, 0);
        Debug.Log(" 1 1 " + result[0]);

        result = Train(1, 0, 1);
        Debug.Log(" 1 0 " + result[0]);

        result = Train(0, 1, 1);
        Debug.Log(" 0 1 " + result[0]);

        result = Train(0, 0, 0);
        Debug.Log(" 0 0 " + result[0]);
    }

    // the Train method, input1, input2, expected output
    List<double> Train(double il, double i2, double o)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (ann.Go(inputs, outputs));
    }

}

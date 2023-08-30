package com.example.myapplication;

import android.app.AlertDialog;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.DelegateFactory;


import java.io.IOException;
import java.util.Arrays;


public class Inference extends Service {
    private AlertDialog alertDialog;

    public Inference() {
    }

    private Interpreter tflite;

    public void TFLiteInference(String MODEL_PATH) throws IOException {
        CompatibilityList compatList = new CompatibilityList();
        Interpreter.Options gpu_options = new Interpreter.Options();
        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate gpuDelegate = new GpuDelegate();
            gpu_options.addDelegate(gpuDelegate);
        }

        // Define the dimensions of the input tensor
        int inputWidth = 1249;
        int inputHeight = 126;
        int inputChannels = 20;

        float[][][][] inputArray = new float[1][inputHeight][inputWidth][inputChannels];

        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                for (int k = 0; k < inputChannels; k++) {
                    inputArray[0][i][j][k] = 0.0f;
                }
            }
        }

        int outputSize = 50; // Assuming the output size is 5
        float[][] outputArray = new float[1][outputSize];


        float[][][][] middleArray = new float[1][124][1247][20];

        Interpreter all_tflite_gpu = new Interpreter(FileUtil.loadMappedFile(Inference.this, "CNN_original.tflite"), gpu_options);
        Interpreter tflite_cpu = new Interpreter(FileUtil.loadMappedFile(Inference.this,"model_part_1.tflite"));
        Interpreter tflite_gpu = new Interpreter(FileUtil.loadMappedFile(Inference.this, "model_part_2.tflite"), gpu_options);
        Interpreter all_tflite_cpu = new Interpreter(FileUtil.loadMappedFile(Inference.this, "CNN_original.tflite") );

        // run on total cpu first time
        long t5 = System.currentTimeMillis();
        all_tflite_cpu.run(inputArray, outputArray);
        long t6 = System.currentTimeMillis();
        Log.d("TAG", "first all_cpu prediction takes " + (t6 - t5) + " ms");
        Log.d("TAG", Arrays.toString(outputArray[0]));

        // run on total cpu second time
        long t7 = System.currentTimeMillis();
        all_tflite_cpu.run(inputArray, outputArray);
        long t8 = System.currentTimeMillis();
        Log.d("TAG", "second all_cpu prediction takes " + (t8 - t7) + " ms");
        Log.d("TAG", Arrays.toString(outputArray[0]));

        // run on total gpu first time
        long t1 = System.currentTimeMillis();
        all_tflite_gpu.run(inputArray, outputArray);
        long t2 = System.currentTimeMillis();
        Log.d("TAG", "first all_gpu prediction takes " + (t2 - t1) + " ms");
        Log.d("TAG", Arrays.toString(outputArray[0]));

        // run on total gpu second time
        long t9= System.currentTimeMillis();
        all_tflite_gpu.run(inputArray, outputArray);
        long t10 = System.currentTimeMillis();
        Log.d("TAG", "second all_gpu prediction takes " + (t10 - t9) + " ms");
        Log.d("TAG", Arrays.toString(outputArray[0]));

        // run on cpu and gpu first time
        long t3 = System.currentTimeMillis();
        tflite_cpu.run(inputArray, middleArray);
        tflite_gpu.run(middleArray, outputArray);
        long t4 = System.currentTimeMillis();
        Log.d("TAG", "partition prediction takes " + (t4 - t3) + " ms");
        Log.d("TAG", Arrays.toString(outputArray[0]));

        // run on cpu and gpu second time
        long t11 = System.currentTimeMillis();
        tflite_cpu.run(inputArray, middleArray);
        tflite_gpu.run(middleArray, outputArray);
        long t12 = System.currentTimeMillis();
        Log.d("TAG", "partition prediction takes " + (t12 - t11) + " ms");
        Log.d("TAG", Arrays.toString(outputArray[0]));

        tflite_cpu.close();
        tflite_gpu.close();
        all_tflite_cpu.close();
        all_tflite_gpu.close();
    }

    @Override
    public void onDestroy() {
        if (alertDialog != null && alertDialog.isShowing()) {
            alertDialog.dismiss();
        }
        super.onDestroy();
    }

    private void showToast() {
        Toast.makeText(this, "Service is running", Toast.LENGTH_SHORT).show();
    }
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        showToast();
        try {
            TFLiteInference("model.tflite");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
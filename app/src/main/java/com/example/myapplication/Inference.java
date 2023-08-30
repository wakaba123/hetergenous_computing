package com.example.myapplication;

import android.app.AlertDialog;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.os.IBinder;
import android.util.Log;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.gpu.GpuDelegate;


import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Objects;


public class Inference extends Service {
    private AlertDialog alertDialog;

    public Inference() {
    }

    private Interpreter tflite;

    public static String[] getAssetFiles(Context context, String folderPath) {
        AssetManager assetManager = context.getAssets();
        String[] files = null;
        try {
            files = assetManager.list(folderPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return files;
    }

    public void TFLiteInference(String MODEL_PATH) throws IOException {
        CompatibilityList compatList = new CompatibilityList();
        Interpreter.Options gpu_options = new Interpreter.Options();
        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate gpuDelegate = new GpuDelegate();
            gpu_options.addDelegate(gpuDelegate);
        }

        String[] models = getAssetFiles(Inference.this, "");

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
        String TAG = "tfliteinference";

        Interpreter all_tflite_gpu = new Interpreter(FileUtil.loadMappedFile(Inference.this, "CNN_original.tflite"), gpu_options);
        Interpreter all_tflite_cpu = new Interpreter(FileUtil.loadMappedFile(Inference.this, "CNN_original.tflite") );


        // run on total cpu first time
        long t5 = System.currentTimeMillis();
        all_tflite_cpu.run(inputArray, outputArray);
        long t6 = System.currentTimeMillis();
        Log.d(TAG, "first all_cpu prediction takes " + (t6 - t5) + " ms");
        Log.d(TAG, Arrays.toString(outputArray[0]));

        // run on total cpu second time
        long t7 = System.currentTimeMillis();
        all_tflite_cpu.run(inputArray, outputArray);
        long t8 = System.currentTimeMillis();
        Log.d(TAG, "second all_cpu prediction takes " + (t8 - t7) + " ms");
        Log.d(TAG, Arrays.toString(outputArray[0]));

        // run on total gpu first time
        long t1 = System.currentTimeMillis();
        all_tflite_gpu.run(inputArray, outputArray);
        long t2 = System.currentTimeMillis();
        Log.d(TAG, "first all_gpu prediction takes " + (t2 - t1) + " ms");
        Log.d(TAG, Arrays.toString(outputArray[0]));

        // run on total gpu second time
        long t9= System.currentTimeMillis();
        all_tflite_gpu.run(inputArray, outputArray);
        long t10 = System.currentTimeMillis();
        Log.d(TAG, "second all_gpu prediction takes " + (t10 - t9) + " ms");
        Log.d(TAG, Arrays.toString(outputArray[0]));

        all_tflite_cpu.close();
        all_tflite_gpu.close();

        // here start partition inference
        int i = 0 ;
        while(i < models.length){
            String[] parts = models[i].split("_");
            if(parts.length < 3){ // not the cutted model file
                i++;
                continue;
            }

            // Extract the cut number
            int cutNumber = Integer.parseInt(parts[2]);

            // Extract the last three numbers
            int lastNumber1 = Integer.parseInt(parts[parts.length - 3]);
            int lastNumber2 = Integer.parseInt(parts[parts.length - 2]);
            int lastNumber3 = Integer.parseInt(parts[parts.length - 1].replace(".tflite", ""));
            Log.d(TAG,"here is cut " + cutNumber);
//            Log.d(TAG,"here is  " + cutNumber + " " + lastNumber1 + " " + lastNumber2 + " " + lastNumber3);

            float[][][][] middleArray = new float[1][lastNumber1][lastNumber2][lastNumber3];

            Interpreter tflite_cpu = new Interpreter(FileUtil.loadMappedFile(Inference.this,models[i]));
            Interpreter tflite_gpu = new Interpreter(FileUtil.loadMappedFile(Inference.this,models[i+1]), gpu_options);

            // run on cpu and gpu first time
            long t3 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            tflite_gpu.run(middleArray, outputArray);
            long t4 = System.currentTimeMillis();
            Log.d(TAG, "first partition prediction takes " + (t4 - t3) + " ms");
//            Log.d(TAG, Arrays.toString(outputArray[0]));

            // run on cpu and gpu second time
            long t11 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            tflite_gpu.run(middleArray, outputArray);
            long t12 = System.currentTimeMillis();
            Log.d(TAG, "second partition prediction takes " + (t12 - t11) + " ms");
//            Log.d(TAG, Arrays.toString(outputArray[0]));

            tflite_cpu.close();
            tflite_gpu.close();
            i += 2;
        }
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
#include<stdio.h>
#include<malloc.h>


void vecAdd(float* a, float* b, float* out, int len){
    for(int i=0; i<len; i++){
        out[i] = a[i] + b[i];
    }
}


int main(){
    float a[3] = {1.0, 2.0, 3.0};
    float b[3] = {2.0, 3.0, 4.0};
    //float c[3] = {0.0, 0.0, 0.0};
    float* c = malloc(__SIZEOF_FLOAT__ * 3);
    vecAdd(a, b, c, 3);
    for (int i = 0; i < 3; i++)
    {
        printf("%f, ", c[i]);
    }
    free(c);
    printf("\n");
}
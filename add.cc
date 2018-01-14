int main()
{
    float x[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float y[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float z[16];
    
    float sum = 0;
     __asm__("#loop start");
    for (int i =0; i < 16; ++i)
    {
        z[i]=x[i]+y[i];
    }
     __asm__("#loop end");
    for (int i =0; i < 16; ++i)
    {
        sum+=z[i];
    }
    return sum;
}

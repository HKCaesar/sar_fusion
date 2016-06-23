//
// Created by auroua on 16-6-23.
//

#include <iostream>
using namespace std;

int recursive(int i){
    int sum = 0;
    if(0==i){
        return 1;
    }else{
        sum = i*recursive(i-1);
    }
}

int main(){
    int value = 100;
    int result = recursive(value);
    cout<<result;
    return result;
}

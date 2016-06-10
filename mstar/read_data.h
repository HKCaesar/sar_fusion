//
// Created by auroua on 16-6-10.
//

#ifndef SAR_FUSION_READ_DATA_H
#define SAR_FUSION_READ_DATA_H
unsigned char*    read_mstar(const char* MSTARname, const char* JPEGname, int HDRflag, int ENHANCEflag, int VERBOSEflag, int qfactor);
int       CheckByteOrder();
float     byteswap_SR_IR(unsigned char *pointer);
unsigned short byteswap_SUS_IUS(unsigned char *pointer);

#endif //SAR_FUSION_READ_DATA_H

// MIT License
// 
// Copyright (c) 2021 PingzhouMing
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include "io/my_file.h"

int main(void)
{
    int x1, x2, y;
    char ch;
    char s1[100]={}, s2[100]={};
    my_file_t *fr;
    my_file_t *fw;

    fw = my_fopen("test.txt", "w");
    my_fprintf(fw, "This %c%c my %dst try: %s .\n", 'i', 's', 1, "Hello World!!");
    my_fprintf(fw, "Mathematical excercise\n");
    my_fprintf(fw, "for the %c %s\n", '+', "operation");
    my_fprintf(fw, "\t%d+%d= %d\n", -99, 77, 77-99);
	my_fwrite(fw, "test.txt", 1, strlen("test.txt"));
    my_fprintf(fw, "\n");
    my_fclose(fw);

    fr = my_fopen(argv[1], "r");

	my_fread(s1, sizeof(char), 36, fr);
    printf("Preamble: %s\n", s1);

    my_fscanf(fr, "%s", s1); // "Mathematical"
    my_fscanf(fr, "%s", s2); // "excercise"
	printf("The title is [%s %s]\n", s1, s2);

    my_fscanf(fr, "%s", s1); // "for"
    my_fscanf(fr, "%s", s2); // "the"
    my_fscanf(fr, "%c", &ch); // ' '
    my_fscanf(fr, "%c", &ch); // '+'
    printf("The operation is '[%c]'\n", ch);

    my_fscanf(fr, "%s", s2); // "operation"
	ch = 'm';
    my_fscanf(fr, "%d", &x1);
    my_fscanf(fr, "%c", &ch); // '+'
    my_fscanf(fr, "%d", &x2);
    my_fscanf(fr, "%c", &ch); // '='
    my_fscanf(fr, "%d", &y);
    printf("x1=[%d] x2=[%d] y=[%d]\n", x1, x2, y);

    my_fclose(fr);

    return 0;
}

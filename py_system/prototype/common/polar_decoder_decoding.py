#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import math
import numpy as np
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from polarDecode import polarDecode

Q_MAX = np.array([0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20, 256, 34, 24, 36, 7, 
                  129, 66, 512, 11, 40, 68, 130, 19, 13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 
                  25, 22, 136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49, 74, 272, 160, 520, 288, 
                  528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52, 23, 134, 384, 76, 137, 82, 56, 27, 97, 
                  39, 259, 84, 138, 145, 261, 29, 43, 98, 515, 88, 140, 30, 146, 71, 262, 265, 161, 576, 45, 100, 
                  640, 51, 148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77, 164, 768, 268, 274, 518, 54, 
                  83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168, 139, 99, 86, 60, 280, 89, 290, 529, 
                  524, 196, 141, 101, 147, 176, 142, 530, 321, 31, 200, 90, 545, 292, 322, 532, 263, 149, 102, 105, 
                  304, 296, 163, 92, 47, 267, 385, 546, 324, 208, 386, 150, 153, 165, 106, 55, 328, 536, 577, 548, 
                  113, 154, 79, 269, 108, 578, 224, 166, 519, 552, 195, 270, 641, 523, 275, 580, 291, 59, 169, 560, 
                  114, 277, 156, 87, 197, 116, 170, 61, 531, 525, 642, 281, 278, 526, 177, 293, 388, 91, 584, 769, 
                  198, 172, 120, 201, 336, 62, 282, 143, 103, 178, 294, 93, 644, 202, 592, 323, 392, 297, 770, 107, 
                  180, 151, 209, 284, 648, 94, 204, 298, 400, 608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184, 
                  534, 537, 115, 167, 225, 326, 306, 772, 157, 656, 329, 110, 117, 212, 171, 776, 330, 226, 549, 538, 
                  387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332, 579, 540, 389, 173, 121, 553, 199, 784, 
                  179, 228, 338, 312, 704, 390, 174, 554, 581, 393, 283, 122, 448, 353, 561, 203, 63, 340, 394, 527, 
                  582, 556, 181, 295, 285, 232, 124, 205, 182, 643, 562, 286, 585, 299, 354, 211, 401, 185, 396, 344, 
                  586, 645, 593, 535, 240, 206, 95, 327, 564, 800, 402, 356, 307, 301, 417, 213, 568, 832, 588, 186, 
                  646, 404, 227, 896, 594, 418, 302, 649, 771, 360, 539, 111, 331, 214, 309, 188, 449, 217, 408, 609, 
                  596, 551, 650, 229, 159, 420, 310, 541, 773, 610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 
                  313, 450, 542, 334, 233, 555, 774, 175, 123, 658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 
                  287, 183, 234, 125, 557, 660, 616, 342, 316, 241, 778, 563, 345, 452, 397, 403, 207, 674, 558, 785, 
                  432, 357, 187, 236, 664, 624, 587, 780, 705, 126, 242, 565, 398, 346, 456, 358, 405, 303, 569, 244, 
                  595, 189, 566, 676, 361, 706, 589, 215, 786, 647, 348, 419, 406, 464, 680, 801, 362, 590, 409, 570, 
                  788, 597, 572, 219, 311, 708, 598, 601, 651, 421, 792, 802, 611, 602, 410, 231, 688, 653, 248, 369, 
                  190, 364, 654, 659, 335, 480, 315, 221, 370, 613, 422, 425, 451, 614, 543, 235, 412, 343, 372, 775, 
                  317, 222, 426, 453, 237, 559, 833, 804, 712, 834, 661, 808, 779, 617, 604, 433, 720, 816, 836, 347, 
                  897, 243, 662, 454, 318, 675, 618, 898, 781, 376, 428, 665, 736, 567, 840, 625, 238, 359, 457, 399, 
                  787, 591, 678, 434, 677, 349, 245, 458, 666, 620, 363, 127, 191, 782, 407, 436, 626, 571, 465, 681, 
                  246, 707, 350, 599, 668, 790, 460, 249, 682, 573, 411, 803, 789, 709, 365, 440, 628, 689, 374, 423, 
                  466, 793, 250, 371, 481, 574, 413, 603, 366, 468, 655, 900, 805, 615, 684, 710, 429, 794, 252, 373, 
                  605, 848, 690, 713, 632, 482, 806, 427, 904, 414, 223, 663, 692, 835, 619, 472, 455, 796, 809, 714, 
                  721, 837, 716, 864, 810, 606, 912, 722, 696, 377, 435, 817, 319, 621, 812, 484, 430, 838, 667, 488, 
                  239, 378, 459, 622, 627, 437, 380, 818, 461, 496, 669, 679, 724, 841, 629, 351, 467, 438, 737, 251, 
                  462, 442, 441, 469, 247, 683, 842, 738, 899, 670, 783, 849, 820, 728, 928, 791, 367, 901, 630, 685, 
                  844, 633, 711, 253, 691, 824, 902, 686, 740, 850, 375, 444, 470, 483, 415, 485, 905, 795, 473, 634, 
                  744, 852, 960, 865, 693, 797, 906, 715, 807, 474, 636, 694, 254, 717, 575, 913, 798, 811, 379, 697, 
                  431, 607, 489, 866, 723, 486, 908, 718, 813, 476, 856, 839, 725, 698, 914, 752, 868, 819, 814, 439, 
                  929, 490, 623, 671, 739, 916, 463, 843, 381, 497, 930, 821, 726, 961, 872, 492, 631, 729, 700, 443, 
                  741, 845, 920, 382, 822, 851, 730, 498, 880, 742, 445, 471, 635, 932, 687, 903, 825, 500, 846, 745, 
                  826, 732, 446, 962, 936, 475, 853, 867, 637, 907, 487, 695, 746, 828, 753, 854, 857, 504, 799, 255, 
                  964, 909, 719, 477, 915, 638, 748, 944, 869, 491, 699, 754, 858, 478, 968, 383, 910, 815, 976, 870, 
                  917, 727, 493, 873, 701, 931, 756, 860, 499, 731, 823, 922, 874, 918, 502, 933, 743, 760, 881, 494, 
                  702, 921, 501, 876, 847, 992, 447, 733, 827, 934, 882, 937, 963, 747, 505, 855, 924, 734, 829, 965, 
                  938, 884, 506, 749, 945, 966, 755, 859, 940, 830, 911, 871, 639, 888, 479, 946, 750, 969, 508, 861, 
                  757, 970, 919, 875, 862, 758, 948, 977, 923, 972, 761, 877, 952, 495, 703, 935, 978, 883, 762, 503, 
                  925, 878, 735, 993, 885, 939, 994, 980, 926, 764, 941, 967, 886, 831, 947, 507, 889, 984, 751, 942, 
                  996, 971, 890, 509, 949, 973, 1000, 892, 950, 863, 759, 1008, 510, 979, 953, 763, 974, 954, 879, 981, 
                  982, 927, 995, 765, 956, 887, 985, 997, 986, 943, 891, 998, 766, 511, 988, 1001, 951, 1002, 893, 975, 
                  894, 1009, 955, 1004, 1010, 957, 983, 958, 987, 1012, 999, 1016, 767, 989, 1003, 990, 1005, 959, 1011, 
                  1013, 895, 1006, 1014, 1017, 1018, 991, 1020, 1007, 1015, 1019, 1021, 1022, 1023])

def get_n(k, e, n_max):
    """
    TS 38.212 section 5.3.1
    """
    cl2e = math.ceil(math.log2(e))
    if (e <= (9/8) * 2**(cl2e - 1)) and (k / e < 9 / 16):
        n1 = cl2e - 1
    else:
        n1 = cl2e
    r_min = 1 / 8
    n2 = math.ceil(math.log2(k / r_min))
    n_min = 5
    n = max(min(n1, n2, n_max), n_min)
    return 2**n

def get_rate_matching_pattern(k, n, e):
    pi = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13,
                   21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31])
    d = np.arange(0, n, 1)
    jn = np.zeros((1, n), dtype=int)
    yn = np.zeros((1, n), dtype=int)
    for i in range(n):
        temp = math.floor(32 * i / n)
        jn[0][i] = pi[temp] * (n / 32) + (i % (n / 32))
        yn[0][i] = d[jn[0][i]]
    rate_matching = np.zeros((1, e), dtype=int)
    if e >= n:
        for i in range(e):
            rate_matching[0][i] = yn[0][i % n]
            mode = 'repetition'
    else:
        if k / e <= 7/16:
            for i in range(e):
                rate_matching[0][i] = yn[0][i + n - e]
                mode = 'puncturing'
        else:
            for i in range(e):
                rate_matching[0][i] = yn[0][i]
                mode = 'shortening'
    return rate_matching, mode

def get_sequence_pattern(n):
    if n > len(Q_MAX):
        print('Value of n is un-supported !')
    n_qmax = np.array(Q_MAX)
    q_n = n_qmax[n_qmax < n]
    return q_n

def get_channel_interleave(e):
    """
    TS 38.212 section 5.4.1.3
    """
    t = 0
    k = 0
    while t*(t+1)/2 < e:
        t += 1
    vtab = np.zeros((t, t))
    for i in range(t):
        for j in range(t - i):
            if k < e:
                vtab[i, j] = k + 1
            k += 1
    channel_interleave = np.zeros((1, e), dtype=int)
    k = 0
    for j in range(t):
        for i in range(t - j):
            if vtab[i, j] != 0:
                channel_interleave[0][k] = vtab[i, j]
                k += 1
    channel_interleave -= 1
    return channel_interleave

def get_info_bit_pattern(scalar, q_n, rate_matching, mode):
    length = q_n.size
    e = rate_matching.size
    if scalar > length or scalar > e:
        print('something un-supported in information bit')
    temp = np.arange(0, length, 1)
    q_f_n = np.setdiff1d(temp, rate_matching)
    if mode == 'punturing':
        if e >= 3 / 4 * length:
            t = np.arange(0, math.ceil(3 * length/4 - e / 2))
            q_f_n = np.r_[q_f_n, t]
        else:
            t = np.arange(0, math.ceil(9*length/16 - e / 4))
            q_f_n = np.r_[q_f_n, t]
    q_i_n_temp = np.setdiff1d(q_n, q_f_n, assume_unique=True)
    q_i_n = q_i_n_temp[-scalar:]
    info_bit = np.zeros((1, length))
    info_bit[0][q_i_n] = 1
    return info_bit

def polar_decoder_decoding(llr, A, L=1):
    """
    Parameters
    ----------
    llr :
    A   :
    modulation :
    L :
    Returns
    -------
    None.
    """
    len_polar = np.size(llr)
    if A < 12:
        print('No supported length of Polar decoding !')
        return False
    elif A <= 19:
        crc_pattern = np.array([1, 1, 0, 0, 0, 0, 1])
    else:
        crc_pattern = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        if (A >= 1013 or (A >= 360 and len_polar >= 1088)):
            c = 2
        else:
            c = 1
    len_crc = np.size(crc_pattern) - 1
    k = math.ceil(A/c) + len_crc
    e_r = math.floor(len_polar / c)
    if e_r > 8192 or e_r <= 21:
        return False
    
    n_max = 10
    n = get_n(k, e_r, n_max)
    # Get rate-matching pattern
    rate_match_group = get_rate_matching_pattern(k, n, e_r)
    # Get seq pattern
    q_n = get_sequence_pattern(n)
    # Get channel inter-leaving
    channel_inter = get_channel_interleave(e_r)

    a_hat = []
    if A <= 19:
        print('No supported length of Polar decoding for 12 - 19 !')
        return False
    else:
        info_bit = get_info_bit_pattern(k, q_n, rate_match_group[0], rate_match_group[1])
        if c == 2:
            info_bit_part2 = np.copy(info_bit)
            if (A % 2 == 1):
                temp = info_bit.reshape((info_bit.size, ), order='F')
                index = np.where(temp == 1)
                info_bit_part2[index[0][0]] = 0
            e_tilde = np.zeros((1, e_r))
            e_tilde[channel_inter] = llr[0:e_r]
            polar_decode = polarDecode(e_tilde, crc_pattern)
            a_hat = polar_decode.ca_polar_decode(rate_match_group[1], rate_match_group[0], info_bit_part2, L, 3)
            
            if a_hat.size == math.floor(A / c):
                e_tilde[channel_inter] = llr[e_r:2*e_r]
                a2_hat = polar_decode.ca_polar_decode(rate_match_group[1], rate_match_group[0], info_bit, L, 3)
                a_hat = np.r_[a_hat, a2_hat]
            if a_hat.size != A:
                a_hat = np.array([])
        else:
            e_tilde = np.zeros((1, len_polar), dtype=int)
            e_tilde[0][channel_inter] = llr
            polar_decode = polarDecode(e_tilde, crc_pattern)
            a_hat = polar_decode.ca_polar_decode(rate_match_group[1], rate_match_group[0], info_bit, L, 3)
    return a_hat


if __name__ == '__main__':
    print('unit test')
    llr = np.array([-32, 24, -16, -30, -11, 4, 4, -32, -31, 5, -21, -19, -32, -31, -17,
                    -14, -13, -16, 10, -10, -31, 31, 18, -18, -31, -32, 9, 19, -19,
                    -10, -32, 14, 10, -31, 16, -12, 31, -31, 11, -32, -18, 22, -21,
                    31, 31, 16, -10, 14])
    print(np.size(llr))
    print(polar_decoder_decoding(llr, 23))
    llr = np.array([-31, 31, -16, 7, 2, -6, 19, -31, -21, -31,
                    -2, 11, -31, -5, 7, 21, -7, 9, 31, 31,
                    -19, -24, 1, 3, 19, 31, -5, -28, -6, -17,
                    -31, -31, 1, 12, -15, 6, 27, 2, 1, -24,
                    -12, -11, -2, 31, -31, -6, 19, -15, -31, 31,
                    -21, -11, 7, -3, -31, -31, 9, -31, -11, -31,
                    -15, 31, 8, 31, -2, -20, 31, -13, 13, 21,
                    5, 4, -31, 12, 14, -8, -3, 2, 31, 31,
                    17, -31, 0, 20, -31, -14, -10, -13, -3, 0,
                    -19, 22, -22, 19, 3, -1, -31, 1, -23, -17,
                    13, 9, -12, 28, -20, -2, -5, 12, 27, -31,
                    6, -10, -4, -1, -31, 25, 7, -10, 11, 7,
                    11, -31, -20, 0, -6, 15, 31, 31, 31, -16,
                    -24, -8, -31, 23, -9, 10, 7, -5, -31, 6,
                    15, 31, -11, -21, -29, -9, -2, 16, -11, 4,
                    13, -2, 28, -31, -9, -20, -31, 1, -13, -22,
                    1, 11, -31, 31, 31, -26, 15, -6])
    print(np.size(llr))
    print(polar_decoder_decoding(llr, 23))


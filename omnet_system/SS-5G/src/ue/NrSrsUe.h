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

#ifndef UE_NRSRSUE_H_
#define UE_NRSRSUE_H_

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include <vector>
#include "../message/airframe_m.h"
#include "../message/bscontrol_m.h"
#include "../NrEntity.h"
#include "NrUeBase.h"

namespace ss5G {

using namespace omnetpp;

class NrSrsUe : public NrUeBase
{
  private:
    double ue2bs_dist[4];
    int bs_connected;
    cMessage *srs_control_event;

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

    void update_positions();
    virtual void srsPeriodForward();

    bool check_state();
};

};

#endif /* UE_NRSRSUE_H_ */

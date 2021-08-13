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

#ifndef APP_NRPOSITION_H_
#define APP_NRPOSITION_H_

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include <vector>
#include "../message/airframe_m.h"
#include "../message/bscontrol_m.h"
#include "../NrEntity.h"
#include "../gnb/NrGnbBase.h"

namespace ss5G {

using namespace omnetpp;

class NrPosition : public NrGnbBase
{
  private:
    long numSent;
    long numReceived;
    double period_positioning;
    cMessage *position_request_msg;

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

    // Send positioning request
    virtual void forward2core_positioning_request(AirFrameMsg *ttmsg_ue);
};

};

#endif /* APP_NRPOSITION_H_ */
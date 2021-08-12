//
// Generated file, do not edit! Created by nedtool 5.6 from corepacket.msg.
//

#ifndef __COREPACKET_M_H
#define __COREPACKET_M_H

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wreserved-id-macro"
#endif
#include <omnetpp.h>

// nedtool version check
#define MSGC_VERSION 0x0506
#if (MSGC_VERSION!=OMNETPP_VERSION)
#    error Version mismatch! Probably this file was generated by an earlier version of nedtool: 'make clean' should help.
#endif



/**
 * Class generated from <tt>corepacket.msg:19</tt> by nedtool.
 * <pre>
 * //
 * // TODO generated message class
 * //
 * packet CorePacket
 * {
 *     int type;
 * }
 * </pre>
 */
class CorePacket : public ::omnetpp::cPacket
{
  protected:
    int type;

  private:
    void copy(const CorePacket& other);

  protected:
    // protected and unimplemented operator==(), to prevent accidental usage
    bool operator==(const CorePacket&);

  public:
    CorePacket(const char *name=nullptr, short kind=0);
    CorePacket(const CorePacket& other);
    virtual ~CorePacket();
    CorePacket& operator=(const CorePacket& other);
    virtual CorePacket *dup() const override {return new CorePacket(*this);}
    virtual void parsimPack(omnetpp::cCommBuffer *b) const override;
    virtual void parsimUnpack(omnetpp::cCommBuffer *b) override;

    // field getter/setter methods
    virtual int getType() const;
    virtual void setType(int type);
};

inline void doParsimPacking(omnetpp::cCommBuffer *b, const CorePacket& obj) {obj.parsimPack(b);}
inline void doParsimUnpacking(omnetpp::cCommBuffer *b, CorePacket& obj) {obj.parsimUnpack(b);}


#endif // ifndef __COREPACKET_M_H


//
// Generated file, do not edit! Created by nedtool 5.6 from message/airframe.msg.
//

// Disable warnings about unused variables, empty switch stmts, etc:
#ifdef _MSC_VER
#  pragma warning(disable:4101)
#  pragma warning(disable:4065)
#endif

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wshadow"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wc++98-compat"
#  pragma clang diagnostic ignored "-Wunreachable-code-break"
#  pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#include <iostream>
#include <sstream>
#include "airframe_m.h"

namespace omnetpp {

// Template pack/unpack rules. They are declared *after* a1l type-specific pack functions for multiple reasons.
// They are in the omnetpp namespace, to allow them to be found by argument-dependent lookup via the cCommBuffer argument

// Packing/unpacking an std::vector
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::vector<T,A>& v)
{
    int n = v.size();
    doParsimPacking(buffer, n);
    for (int i = 0; i < n; i++)
        doParsimPacking(buffer, v[i]);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::vector<T,A>& v)
{
    int n;
    doParsimUnpacking(buffer, n);
    v.resize(n);
    for (int i = 0; i < n; i++)
        doParsimUnpacking(buffer, v[i]);
}

// Packing/unpacking an std::list
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::list<T,A>& l)
{
    doParsimPacking(buffer, (int)l.size());
    for (typename std::list<T,A>::const_iterator it = l.begin(); it != l.end(); ++it)
        doParsimPacking(buffer, (T&)*it);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::list<T,A>& l)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        l.push_back(T());
        doParsimUnpacking(buffer, l.back());
    }
}

// Packing/unpacking an std::set
template<typename T, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::set<T,Tr,A>& s)
{
    doParsimPacking(buffer, (int)s.size());
    for (typename std::set<T,Tr,A>::const_iterator it = s.begin(); it != s.end(); ++it)
        doParsimPacking(buffer, *it);
}

template<typename T, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::set<T,Tr,A>& s)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        T x;
        doParsimUnpacking(buffer, x);
        s.insert(x);
    }
}

// Packing/unpacking an std::map
template<typename K, typename V, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::map<K,V,Tr,A>& m)
{
    doParsimPacking(buffer, (int)m.size());
    for (typename std::map<K,V,Tr,A>::const_iterator it = m.begin(); it != m.end(); ++it) {
        doParsimPacking(buffer, it->first);
        doParsimPacking(buffer, it->second);
    }
}

template<typename K, typename V, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::map<K,V,Tr,A>& m)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        K k; V v;
        doParsimUnpacking(buffer, k);
        doParsimUnpacking(buffer, v);
        m[k] = v;
    }
}

// Default pack/unpack function for arrays
template<typename T>
void doParsimArrayPacking(omnetpp::cCommBuffer *b, const T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimPacking(b, t[i]);
}

template<typename T>
void doParsimArrayUnpacking(omnetpp::cCommBuffer *b, T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimUnpacking(b, t[i]);
}

// Default rule to prevent compiler from choosing base class' doParsimPacking() function
template<typename T>
void doParsimPacking(omnetpp::cCommBuffer *, const T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimPacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

template<typename T>
void doParsimUnpacking(omnetpp::cCommBuffer *, T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimUnpacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

}  // namespace omnetpp


// forward
template<typename T, typename A>
std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec);

// Template rule which fires if a struct or class doesn't have operator<<
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const T&) {return out;}

// operator<< for std::vector<T>
template<typename T, typename A>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec)
{
    out.put('{');
    for(typename std::vector<T,A>::const_iterator it = vec.begin(); it != vec.end(); ++it)
    {
        if (it != vec.begin()) {
            out.put(','); out.put(' ');
        }
        out << *it;
    }
    out.put('}');
    
    char buf[32];
    sprintf(buf, " (size=%u)", (unsigned int)vec.size());
    out.write(buf, strlen(buf));
    return out;
}

Register_Class(AirFrameMsg)

AirFrameMsg::AirFrameMsg(const char *name, short kind) : ::omnetpp::cMessage(name,kind)
{
    this->type = 0;
    this->source = 0;
    this->destination = 0;
    this->x = 0;
    this->y = 0;
    this->z = 0;
    this->destX = 0;
    this->destY = 0;
    this->destZ = 0;
    this->timeStamp = 0;
    this->txPowerUpdate = 0;
    this->centerFreq = 0;
}

AirFrameMsg::AirFrameMsg(const AirFrameMsg& other) : ::omnetpp::cMessage(other)
{
    copy(other);
}

AirFrameMsg::~AirFrameMsg()
{
}

AirFrameMsg& AirFrameMsg::operator=(const AirFrameMsg& other)
{
    if (this==&other) return *this;
    ::omnetpp::cMessage::operator=(other);
    copy(other);
    return *this;
}

void AirFrameMsg::copy(const AirFrameMsg& other)
{
    this->type = other.type;
    this->source = other.source;
    this->destination = other.destination;
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    this->destX = other.destX;
    this->destY = other.destY;
    this->destZ = other.destZ;
    this->timeStamp = other.timeStamp;
    this->txPowerUpdate = other.txPowerUpdate;
    this->centerFreq = other.centerFreq;
}

void AirFrameMsg::parsimPack(omnetpp::cCommBuffer *b) const
{
    ::omnetpp::cMessage::parsimPack(b);
    doParsimPacking(b,this->type);
    doParsimPacking(b,this->source);
    doParsimPacking(b,this->destination);
    doParsimPacking(b,this->x);
    doParsimPacking(b,this->y);
    doParsimPacking(b,this->z);
    doParsimPacking(b,this->destX);
    doParsimPacking(b,this->destY);
    doParsimPacking(b,this->destZ);
    doParsimPacking(b,this->timeStamp);
    doParsimPacking(b,this->txPowerUpdate);
    doParsimPacking(b,this->centerFreq);
}

void AirFrameMsg::parsimUnpack(omnetpp::cCommBuffer *b)
{
    ::omnetpp::cMessage::parsimUnpack(b);
    doParsimUnpacking(b,this->type);
    doParsimUnpacking(b,this->source);
    doParsimUnpacking(b,this->destination);
    doParsimUnpacking(b,this->x);
    doParsimUnpacking(b,this->y);
    doParsimUnpacking(b,this->z);
    doParsimUnpacking(b,this->destX);
    doParsimUnpacking(b,this->destY);
    doParsimUnpacking(b,this->destZ);
    doParsimUnpacking(b,this->timeStamp);
    doParsimUnpacking(b,this->txPowerUpdate);
    doParsimUnpacking(b,this->centerFreq);
}

int AirFrameMsg::getType() const
{
    return this->type;
}

void AirFrameMsg::setType(int type)
{
    this->type = type;
}

int AirFrameMsg::getSource() const
{
    return this->source;
}

void AirFrameMsg::setSource(int source)
{
    this->source = source;
}

int AirFrameMsg::getDestination() const
{
    return this->destination;
}

void AirFrameMsg::setDestination(int destination)
{
    this->destination = destination;
}

double AirFrameMsg::getX() const
{
    return this->x;
}

void AirFrameMsg::setX(double x)
{
    this->x = x;
}

double AirFrameMsg::getY() const
{
    return this->y;
}

void AirFrameMsg::setY(double y)
{
    this->y = y;
}

double AirFrameMsg::getZ() const
{
    return this->z;
}

void AirFrameMsg::setZ(double z)
{
    this->z = z;
}

double AirFrameMsg::getDestX() const
{
    return this->destX;
}

void AirFrameMsg::setDestX(double destX)
{
    this->destX = destX;
}

double AirFrameMsg::getDestY() const
{
    return this->destY;
}

void AirFrameMsg::setDestY(double destY)
{
    this->destY = destY;
}

double AirFrameMsg::getDestZ() const
{
    return this->destZ;
}

void AirFrameMsg::setDestZ(double destZ)
{
    this->destZ = destZ;
}

double AirFrameMsg::getTimeStamp() const
{
    return this->timeStamp;
}

void AirFrameMsg::setTimeStamp(double timeStamp)
{
    this->timeStamp = timeStamp;
}

double AirFrameMsg::getTxPowerUpdate() const
{
    return this->txPowerUpdate;
}

void AirFrameMsg::setTxPowerUpdate(double txPowerUpdate)
{
    this->txPowerUpdate = txPowerUpdate;
}

double AirFrameMsg::getCenterFreq() const
{
    return this->centerFreq;
}

void AirFrameMsg::setCenterFreq(double centerFreq)
{
    this->centerFreq = centerFreq;
}

class AirFrameMsgDescriptor : public omnetpp::cClassDescriptor
{
  private:
    mutable const char **propertynames;
  public:
    AirFrameMsgDescriptor();
    virtual ~AirFrameMsgDescriptor();

    virtual bool doesSupport(omnetpp::cObject *obj) const override;
    virtual const char **getPropertyNames() const override;
    virtual const char *getProperty(const char *propertyname) const override;
    virtual int getFieldCount() const override;
    virtual const char *getFieldName(int field) const override;
    virtual int findField(const char *fieldName) const override;
    virtual unsigned int getFieldTypeFlags(int field) const override;
    virtual const char *getFieldTypeString(int field) const override;
    virtual const char **getFieldPropertyNames(int field) const override;
    virtual const char *getFieldProperty(int field, const char *propertyname) const override;
    virtual int getFieldArraySize(void *object, int field) const override;

    virtual const char *getFieldDynamicTypeString(void *object, int field, int i) const override;
    virtual std::string getFieldValueAsString(void *object, int field, int i) const override;
    virtual bool setFieldValueAsString(void *object, int field, int i, const char *value) const override;

    virtual const char *getFieldStructName(int field) const override;
    virtual void *getFieldStructValuePointer(void *object, int field, int i) const override;
};

Register_ClassDescriptor(AirFrameMsgDescriptor)

AirFrameMsgDescriptor::AirFrameMsgDescriptor() : omnetpp::cClassDescriptor("AirFrameMsg", "omnetpp::cMessage")
{
    propertynames = nullptr;
}

AirFrameMsgDescriptor::~AirFrameMsgDescriptor()
{
    delete[] propertynames;
}

bool AirFrameMsgDescriptor::doesSupport(omnetpp::cObject *obj) const
{
    return dynamic_cast<AirFrameMsg *>(obj)!=nullptr;
}

const char **AirFrameMsgDescriptor::getPropertyNames() const
{
    if (!propertynames) {
        static const char *names[] = {  nullptr };
        omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
        const char **basenames = basedesc ? basedesc->getPropertyNames() : nullptr;
        propertynames = mergeLists(basenames, names);
    }
    return propertynames;
}

const char *AirFrameMsgDescriptor::getProperty(const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? basedesc->getProperty(propertyname) : nullptr;
}

int AirFrameMsgDescriptor::getFieldCount() const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? 12+basedesc->getFieldCount() : 12;
}

unsigned int AirFrameMsgDescriptor::getFieldTypeFlags(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeFlags(field);
        field -= basedesc->getFieldCount();
    }
    static unsigned int fieldTypeFlags[] = {
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
    };
    return (field>=0 && field<12) ? fieldTypeFlags[field] : 0;
}

const char *AirFrameMsgDescriptor::getFieldName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldName(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldNames[] = {
        "type",
        "source",
        "destination",
        "x",
        "y",
        "z",
        "destX",
        "destY",
        "destZ",
        "timeStamp",
        "txPowerUpdate",
        "centerFreq",
    };
    return (field>=0 && field<12) ? fieldNames[field] : nullptr;
}

int AirFrameMsgDescriptor::findField(const char *fieldName) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    int base = basedesc ? basedesc->getFieldCount() : 0;
    if (fieldName[0]=='t' && strcmp(fieldName, "type")==0) return base+0;
    if (fieldName[0]=='s' && strcmp(fieldName, "source")==0) return base+1;
    if (fieldName[0]=='d' && strcmp(fieldName, "destination")==0) return base+2;
    if (fieldName[0]=='x' && strcmp(fieldName, "x")==0) return base+3;
    if (fieldName[0]=='y' && strcmp(fieldName, "y")==0) return base+4;
    if (fieldName[0]=='z' && strcmp(fieldName, "z")==0) return base+5;
    if (fieldName[0]=='d' && strcmp(fieldName, "destX")==0) return base+6;
    if (fieldName[0]=='d' && strcmp(fieldName, "destY")==0) return base+7;
    if (fieldName[0]=='d' && strcmp(fieldName, "destZ")==0) return base+8;
    if (fieldName[0]=='t' && strcmp(fieldName, "timeStamp")==0) return base+9;
    if (fieldName[0]=='t' && strcmp(fieldName, "txPowerUpdate")==0) return base+10;
    if (fieldName[0]=='c' && strcmp(fieldName, "centerFreq")==0) return base+11;
    return basedesc ? basedesc->findField(fieldName) : -1;
}

const char *AirFrameMsgDescriptor::getFieldTypeString(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeString(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldTypeStrings[] = {
        "int",
        "int",
        "int",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
    };
    return (field>=0 && field<12) ? fieldTypeStrings[field] : nullptr;
}

const char **AirFrameMsgDescriptor::getFieldPropertyNames(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldPropertyNames(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

const char *AirFrameMsgDescriptor::getFieldProperty(int field, const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldProperty(field, propertyname);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

int AirFrameMsgDescriptor::getFieldArraySize(void *object, int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldArraySize(object, field);
        field -= basedesc->getFieldCount();
    }
    AirFrameMsg *pp = (AirFrameMsg *)object; (void)pp;
    switch (field) {
        default: return 0;
    }
}

const char *AirFrameMsgDescriptor::getFieldDynamicTypeString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldDynamicTypeString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    AirFrameMsg *pp = (AirFrameMsg *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}

std::string AirFrameMsgDescriptor::getFieldValueAsString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldValueAsString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    AirFrameMsg *pp = (AirFrameMsg *)object; (void)pp;
    switch (field) {
        case 0: return long2string(pp->getType());
        case 1: return long2string(pp->getSource());
        case 2: return long2string(pp->getDestination());
        case 3: return double2string(pp->getX());
        case 4: return double2string(pp->getY());
        case 5: return double2string(pp->getZ());
        case 6: return double2string(pp->getDestX());
        case 7: return double2string(pp->getDestY());
        case 8: return double2string(pp->getDestZ());
        case 9: return double2string(pp->getTimeStamp());
        case 10: return double2string(pp->getTxPowerUpdate());
        case 11: return double2string(pp->getCenterFreq());
        default: return "";
    }
}

bool AirFrameMsgDescriptor::setFieldValueAsString(void *object, int field, int i, const char *value) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->setFieldValueAsString(object,field,i,value);
        field -= basedesc->getFieldCount();
    }
    AirFrameMsg *pp = (AirFrameMsg *)object; (void)pp;
    switch (field) {
        case 0: pp->setType(string2long(value)); return true;
        case 1: pp->setSource(string2long(value)); return true;
        case 2: pp->setDestination(string2long(value)); return true;
        case 3: pp->setX(string2double(value)); return true;
        case 4: pp->setY(string2double(value)); return true;
        case 5: pp->setZ(string2double(value)); return true;
        case 6: pp->setDestX(string2double(value)); return true;
        case 7: pp->setDestY(string2double(value)); return true;
        case 8: pp->setDestZ(string2double(value)); return true;
        case 9: pp->setTimeStamp(string2double(value)); return true;
        case 10: pp->setTxPowerUpdate(string2double(value)); return true;
        case 11: pp->setCenterFreq(string2double(value)); return true;
        default: return false;
    }
}

const char *AirFrameMsgDescriptor::getFieldStructName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructName(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    };
}

void *AirFrameMsgDescriptor::getFieldStructValuePointer(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructValuePointer(object, field, i);
        field -= basedesc->getFieldCount();
    }
    AirFrameMsg *pp = (AirFrameMsg *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}



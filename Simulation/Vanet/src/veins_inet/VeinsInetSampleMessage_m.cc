//
// Generated file, do not edit! Created by nedtool 5.6 from veins_inet/VeinsInetSampleMessage.msg.
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
#include <memory>
#include "VeinsInetSampleMessage_m.h"

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
    for (int i = 0; i < n; i++) {
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
    for (int i = 0; i < n; i++) {
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
    for (int i = 0; i < n; i++) {
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

namespace {
template <class T> inline
typename std::enable_if<std::is_polymorphic<T>::value && std::is_base_of<omnetpp::cObject,T>::value, void *>::type
toVoidPtr(T* t)
{
    return (void *)(static_cast<const omnetpp::cObject *>(t));
}

template <class T> inline
typename std::enable_if<std::is_polymorphic<T>::value && !std::is_base_of<omnetpp::cObject,T>::value, void *>::type
toVoidPtr(T* t)
{
    return (void *)dynamic_cast<const void *>(t);
}

template <class T> inline
typename std::enable_if<!std::is_polymorphic<T>::value, void *>::type
toVoidPtr(T* t)
{
    return (void *)static_cast<const void *>(t);
}

}


// forward
template<typename T, typename A>
std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec);

// Template rule to generate operator<< for shared_ptr<T>
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const std::shared_ptr<T>& t) { return out << t.get(); }

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

Register_Class(VeinsInetSampleMessage)

VeinsInetSampleMessage::VeinsInetSampleMessage() : ::inet::FieldsChunk()
{
}

VeinsInetSampleMessage::VeinsInetSampleMessage(const VeinsInetSampleMessage& other) : ::inet::FieldsChunk(other)
{
    copy(other);
}

VeinsInetSampleMessage::~VeinsInetSampleMessage()
{
}

VeinsInetSampleMessage& VeinsInetSampleMessage::operator=(const VeinsInetSampleMessage& other)
{
    if (this == &other) return *this;
    ::inet::FieldsChunk::operator=(other);
    copy(other);
    return *this;
}

void VeinsInetSampleMessage::copy(const VeinsInetSampleMessage& other)
{
    this->senderId = other.senderId;
    this->posx = other.posx;
    this->posy = other.posy;
    this->spdx = other.spdx;
    this->spdy = other.spdy;
    this->aclx = other.aclx;
    this->acly = other.acly;
    this->hedx = other.hedx;
    this->hedy = other.hedy;
    this->malicious = other.malicious;
    this->attackType = other.attackType;
}

void VeinsInetSampleMessage::parsimPack(omnetpp::cCommBuffer *b) const
{
    ::inet::FieldsChunk::parsimPack(b);
    doParsimPacking(b,this->senderId);
    doParsimPacking(b,this->posx);
    doParsimPacking(b,this->posy);
    doParsimPacking(b,this->spdx);
    doParsimPacking(b,this->spdy);
    doParsimPacking(b,this->aclx);
    doParsimPacking(b,this->acly);
    doParsimPacking(b,this->hedx);
    doParsimPacking(b,this->hedy);
    doParsimPacking(b,this->malicious);
    doParsimPacking(b,this->attackType);
}

void VeinsInetSampleMessage::parsimUnpack(omnetpp::cCommBuffer *b)
{
    ::inet::FieldsChunk::parsimUnpack(b);
    doParsimUnpacking(b,this->senderId);
    doParsimUnpacking(b,this->posx);
    doParsimUnpacking(b,this->posy);
    doParsimUnpacking(b,this->spdx);
    doParsimUnpacking(b,this->spdy);
    doParsimUnpacking(b,this->aclx);
    doParsimUnpacking(b,this->acly);
    doParsimUnpacking(b,this->hedx);
    doParsimUnpacking(b,this->hedy);
    doParsimUnpacking(b,this->malicious);
    doParsimUnpacking(b,this->attackType);
}

const char * VeinsInetSampleMessage::getSenderId() const
{
    return this->senderId.c_str();
}

void VeinsInetSampleMessage::setSenderId(const char * senderId)
{
    handleChange();
    this->senderId = senderId;
}

double VeinsInetSampleMessage::getPosx() const
{
    return this->posx;
}

void VeinsInetSampleMessage::setPosx(double posx)
{
    handleChange();
    this->posx = posx;
}

double VeinsInetSampleMessage::getPosy() const
{
    return this->posy;
}

void VeinsInetSampleMessage::setPosy(double posy)
{
    handleChange();
    this->posy = posy;
}

double VeinsInetSampleMessage::getSpdx() const
{
    return this->spdx;
}

void VeinsInetSampleMessage::setSpdx(double spdx)
{
    handleChange();
    this->spdx = spdx;
}

double VeinsInetSampleMessage::getSpdy() const
{
    return this->spdy;
}

void VeinsInetSampleMessage::setSpdy(double spdy)
{
    handleChange();
    this->spdy = spdy;
}

double VeinsInetSampleMessage::getAclx() const
{
    return this->aclx;
}

void VeinsInetSampleMessage::setAclx(double aclx)
{
    handleChange();
    this->aclx = aclx;
}

double VeinsInetSampleMessage::getAcly() const
{
    return this->acly;
}

void VeinsInetSampleMessage::setAcly(double acly)
{
    handleChange();
    this->acly = acly;
}

double VeinsInetSampleMessage::getHedx() const
{
    return this->hedx;
}

void VeinsInetSampleMessage::setHedx(double hedx)
{
    handleChange();
    this->hedx = hedx;
}

double VeinsInetSampleMessage::getHedy() const
{
    return this->hedy;
}

void VeinsInetSampleMessage::setHedy(double hedy)
{
    handleChange();
    this->hedy = hedy;
}

bool VeinsInetSampleMessage::getMalicious() const
{
    return this->malicious;
}

void VeinsInetSampleMessage::setMalicious(bool malicious)
{
    handleChange();
    this->malicious = malicious;
}

const char * VeinsInetSampleMessage::getAttackType() const
{
    return this->attackType.c_str();
}

void VeinsInetSampleMessage::setAttackType(const char * attackType)
{
    handleChange();
    this->attackType = attackType;
}

class VeinsInetSampleMessageDescriptor : public omnetpp::cClassDescriptor
{
  private:
    mutable const char **propertynames;
    enum FieldConstants {
        FIELD_senderId,
        FIELD_posx,
        FIELD_posy,
        FIELD_spdx,
        FIELD_spdy,
        FIELD_aclx,
        FIELD_acly,
        FIELD_hedx,
        FIELD_hedy,
        FIELD_malicious,
        FIELD_attackType,
    };
  public:
    VeinsInetSampleMessageDescriptor();
    virtual ~VeinsInetSampleMessageDescriptor();

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

Register_ClassDescriptor(VeinsInetSampleMessageDescriptor)

VeinsInetSampleMessageDescriptor::VeinsInetSampleMessageDescriptor() : omnetpp::cClassDescriptor(omnetpp::opp_typename(typeid(VeinsInetSampleMessage)), "inet::FieldsChunk")
{
    propertynames = nullptr;
}

VeinsInetSampleMessageDescriptor::~VeinsInetSampleMessageDescriptor()
{
    delete[] propertynames;
}

bool VeinsInetSampleMessageDescriptor::doesSupport(omnetpp::cObject *obj) const
{
    return dynamic_cast<VeinsInetSampleMessage *>(obj)!=nullptr;
}

const char **VeinsInetSampleMessageDescriptor::getPropertyNames() const
{
    if (!propertynames) {
        static const char *names[] = {  nullptr };
        omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
        const char **basenames = basedesc ? basedesc->getPropertyNames() : nullptr;
        propertynames = mergeLists(basenames, names);
    }
    return propertynames;
}

const char *VeinsInetSampleMessageDescriptor::getProperty(const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? basedesc->getProperty(propertyname) : nullptr;
}

int VeinsInetSampleMessageDescriptor::getFieldCount() const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? 11+basedesc->getFieldCount() : 11;
}

unsigned int VeinsInetSampleMessageDescriptor::getFieldTypeFlags(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeFlags(field);
        field -= basedesc->getFieldCount();
    }
    static unsigned int fieldTypeFlags[] = {
        FD_ISEDITABLE,    // FIELD_senderId
        FD_ISEDITABLE,    // FIELD_posx
        FD_ISEDITABLE,    // FIELD_posy
        FD_ISEDITABLE,    // FIELD_spdx
        FD_ISEDITABLE,    // FIELD_spdy
        FD_ISEDITABLE,    // FIELD_aclx
        FD_ISEDITABLE,    // FIELD_acly
        FD_ISEDITABLE,    // FIELD_hedx
        FD_ISEDITABLE,    // FIELD_hedy
        FD_ISEDITABLE,    // FIELD_malicious
        FD_ISEDITABLE,    // FIELD_attackType
    };
    return (field >= 0 && field < 11) ? fieldTypeFlags[field] : 0;
}

const char *VeinsInetSampleMessageDescriptor::getFieldName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldName(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldNames[] = {
        "senderId",
        "posx",
        "posy",
        "spdx",
        "spdy",
        "aclx",
        "acly",
        "hedx",
        "hedy",
        "malicious",
        "attackType",
    };
    return (field >= 0 && field < 11) ? fieldNames[field] : nullptr;
}

int VeinsInetSampleMessageDescriptor::findField(const char *fieldName) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    int base = basedesc ? basedesc->getFieldCount() : 0;
    if (fieldName[0] == 's' && strcmp(fieldName, "senderId") == 0) return base+0;
    if (fieldName[0] == 'p' && strcmp(fieldName, "posx") == 0) return base+1;
    if (fieldName[0] == 'p' && strcmp(fieldName, "posy") == 0) return base+2;
    if (fieldName[0] == 's' && strcmp(fieldName, "spdx") == 0) return base+3;
    if (fieldName[0] == 's' && strcmp(fieldName, "spdy") == 0) return base+4;
    if (fieldName[0] == 'a' && strcmp(fieldName, "aclx") == 0) return base+5;
    if (fieldName[0] == 'a' && strcmp(fieldName, "acly") == 0) return base+6;
    if (fieldName[0] == 'h' && strcmp(fieldName, "hedx") == 0) return base+7;
    if (fieldName[0] == 'h' && strcmp(fieldName, "hedy") == 0) return base+8;
    if (fieldName[0] == 'm' && strcmp(fieldName, "malicious") == 0) return base+9;
    if (fieldName[0] == 'a' && strcmp(fieldName, "attackType") == 0) return base+10;
    return basedesc ? basedesc->findField(fieldName) : -1;
}

const char *VeinsInetSampleMessageDescriptor::getFieldTypeString(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeString(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldTypeStrings[] = {
        "string",    // FIELD_senderId
        "double",    // FIELD_posx
        "double",    // FIELD_posy
        "double",    // FIELD_spdx
        "double",    // FIELD_spdy
        "double",    // FIELD_aclx
        "double",    // FIELD_acly
        "double",    // FIELD_hedx
        "double",    // FIELD_hedy
        "bool",    // FIELD_malicious
        "string",    // FIELD_attackType
    };
    return (field >= 0 && field < 11) ? fieldTypeStrings[field] : nullptr;
}

const char **VeinsInetSampleMessageDescriptor::getFieldPropertyNames(int field) const
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

const char *VeinsInetSampleMessageDescriptor::getFieldProperty(int field, const char *propertyname) const
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

int VeinsInetSampleMessageDescriptor::getFieldArraySize(void *object, int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldArraySize(object, field);
        field -= basedesc->getFieldCount();
    }
    VeinsInetSampleMessage *pp = (VeinsInetSampleMessage *)object; (void)pp;
    switch (field) {
        default: return 0;
    }
}

const char *VeinsInetSampleMessageDescriptor::getFieldDynamicTypeString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldDynamicTypeString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    VeinsInetSampleMessage *pp = (VeinsInetSampleMessage *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}

std::string VeinsInetSampleMessageDescriptor::getFieldValueAsString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldValueAsString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    VeinsInetSampleMessage *pp = (VeinsInetSampleMessage *)object; (void)pp;
    switch (field) {
        case FIELD_senderId: return oppstring2string(pp->getSenderId());
        case FIELD_posx: return double2string(pp->getPosx());
        case FIELD_posy: return double2string(pp->getPosy());
        case FIELD_spdx: return double2string(pp->getSpdx());
        case FIELD_spdy: return double2string(pp->getSpdy());
        case FIELD_aclx: return double2string(pp->getAclx());
        case FIELD_acly: return double2string(pp->getAcly());
        case FIELD_hedx: return double2string(pp->getHedx());
        case FIELD_hedy: return double2string(pp->getHedy());
        case FIELD_malicious: return bool2string(pp->getMalicious());
        case FIELD_attackType: return oppstring2string(pp->getAttackType());
        default: return "";
    }
}

bool VeinsInetSampleMessageDescriptor::setFieldValueAsString(void *object, int field, int i, const char *value) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->setFieldValueAsString(object,field,i,value);
        field -= basedesc->getFieldCount();
    }
    VeinsInetSampleMessage *pp = (VeinsInetSampleMessage *)object; (void)pp;
    switch (field) {
        case FIELD_senderId: pp->setSenderId((value)); return true;
        case FIELD_posx: pp->setPosx(string2double(value)); return true;
        case FIELD_posy: pp->setPosy(string2double(value)); return true;
        case FIELD_spdx: pp->setSpdx(string2double(value)); return true;
        case FIELD_spdy: pp->setSpdy(string2double(value)); return true;
        case FIELD_aclx: pp->setAclx(string2double(value)); return true;
        case FIELD_acly: pp->setAcly(string2double(value)); return true;
        case FIELD_hedx: pp->setHedx(string2double(value)); return true;
        case FIELD_hedy: pp->setHedy(string2double(value)); return true;
        case FIELD_malicious: pp->setMalicious(string2bool(value)); return true;
        case FIELD_attackType: pp->setAttackType((value)); return true;
        default: return false;
    }
}

const char *VeinsInetSampleMessageDescriptor::getFieldStructName(int field) const
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

void *VeinsInetSampleMessageDescriptor::getFieldStructValuePointer(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructValuePointer(object, field, i);
        field -= basedesc->getFieldCount();
    }
    VeinsInetSampleMessage *pp = (VeinsInetSampleMessage *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}


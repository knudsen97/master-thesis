#include "../inc/MotionPlanning.hpp"


MotionPlanning::MotionPlanning(rw::models::SerialDevice::Ptr device, rw::core::Ptr<rw::models::WorkCell> wc, rw::kinematics::State state)
{
    // Set robot and workcell
    this->workCell_ = wc;
    this->robot_ = device;
    this->state_ = state;

    // Create inverse kinematics solver
    this->solver_ = rw::common::ownedPtr(new rw::invkin::ClosedFormIKSolverUR(this->robot_, this->state_));
    
    // Create collision detector
    this->collisionStrategy_ = rwlibs::proximitystrategies::ProximityStrategyYaobi::make();
    this->collisionDetector_ = rw::common::ownedPtr(new rw::proximity::CollisionDetector(this->workCell_, this->collisionStrategy_));

    // Create distance metric
    this->distanceMetric_ = rw::math::MetricFactory::makeEuclidean<rw::math::Q>();

    // Create planner
    this->Qconstraint_ = rw::pathplanning::QConstraint::make(this->collisionDetector_, this->robot_, this->state_);
    this->plannerConstraint_ = rw::pathplanning::PlannerConstraint::make(this->collisionDetector_, this->robot_, this->state_);
    this->sampler_ = rw::pathplanning::QSampler::QSampler::makeConstrained(
        rw::pathplanning::QSampler::QSampler::makeUniform(this->robot_), this->plannerConstraint_.getQConstraintPtr()
    );
    double step_size, resolution = 0.1;
    this->edgeContrain_ = rw::pathplanning::QEdgeConstraint::make(this->Qconstraint_.get(), this->distanceMetric_, resolution);
    this->planner_ = rwlibs::pathplanners::RRTPlanner::makeQToQPlanner(
        this->plannerConstraint_, 
        this->sampler_, 
        this->distanceMetric_, 
        step_size, 
        rwlibs::pathplanners::RRTPlanner::RRTConnect
    );

}

MotionPlanning::~MotionPlanning(){}

void MotionPlanning::setTarget(rw::math::Q targetQ)
{
    this->Qtarget_ = targetQ;
}

void MotionPlanning::setStartQ(rw::math::Q startQ)
{
    this->Qstart = startQ;
}

void MotionPlanning::setSubgoalDistance(double distance)
{
    this->subgoalDistance_ = distance;
}

void MotionPlanning::setWorldTobject(rw::math::Transform3D<> worldTobject)
{
    this->worldTobject_ = worldTobject;
}


rw::math::Q MotionPlanning::getShortestSolutionFromQToQVec(rw::math::Q Q, std::vector<rw::math::Q> QVec)
{
    double distance = this->distanceMetric_->distance(QVec[0], Q);
    double calculatedDistance = 0;
    rw::math::Q Qgoal = QVec[0];
    for (auto q : QVec)
    {
        calculatedDistance = this->distanceMetric_->distance(Q, q);
        if (calculatedDistance < distance)
        {
            distance = calculatedDistance;
            Qgoal = q;
        }
    }
    return Qgoal;
}

rw::trajectory::QPath MotionPlanning::linearInterpolator(rw::trajectory::QPath subgoals, double timestep)
{
    rw::trajectory::QPath linIntPath;
    double linIntTimeStep = 1;
    for(unsigned int i = 0; i < subgoals.size()-1; i++)
    {
        rw::trajectory::LinearInterpolator<rw::math::Q> LinInt(subgoals[i], subgoals[i+1], linIntTimeStep);
        for(double dt = 0.0; dt < linIntTimeStep; dt += timestep)
        {
            linIntPath.push_back(LinInt.x(dt));
        }
    }
    return linIntPath;
}

rw::trajectory::QPath MotionPlanning::calculateSubgoals()
{
    rw::trajectory::QPath path;
    rw::math::Q subgoal;
    if (subgoalDistance_ > 0)
    {
        // Move backwards in the direction of the target
        rw::math::Transform3D<> backoff = rw::math::Transform3D<>(rw::math::Vector3D<>(0, 0, subgoalDistance_));
        rw::math::Transform3D<> worldTSubgoal = this->worldTobject_ * backoff;

        // Inverse kinematics
        InverseKinematics solver(this->robot_, this->workCell_, this->state_);
        double angle_step = 0.1; // increment in roll angle
        double start = -M_PI;
        double end = M_PI;
        bool solutionFound = solver.solve(worldTSubgoal, start, end, angle_step);
        if(!solutionFound){
            std::cerr << "Could not move to subgoal" << std::endl;
            return path;
        }
        auto solutions = solver.getSolutions();
        subgoal = this->getShortestSolutionFromQToQVec(this->Qtarget_, solutions);
    }
    else
    {
        subgoal = this->Qtarget_;
    }

    bool pathFound = this->planner_->query(this->Qstart, subgoal, subgoals_);
    if (!pathFound)
    {
        std::cerr << "Could not find path" << std::endl;
        return path;
    }

    if (subgoalDistance_ > 0)
        subgoals_.push_back(this->Qtarget_);

    return subgoals_;
}

rw::trajectory::QPath MotionPlanning::getLinearPath(double timestep)
{
    this->path_ = this->linearInterpolator(this->subgoals_, timestep);
    return this->path_;
}


/* helper function */
rw::trajectory::TimedStatePath createReplay(rw::models::WorkCell::Ptr wc, rw::trajectory::QPath path, rw::models::SerialDevice::Ptr robot, double dt)
{
    rw::trajectory::TimedStatePath pathReplay;
    auto stateCopy = wc->getDefaultState().clone();
    double time = 0;
    for (auto q : path)
    {
        robot->setQ(q, stateCopy);
        pathReplay.push_back(rw::trajectory::TimedState(time, stateCopy));
        time += dt;
    }    
    return pathReplay;
}


rw::trajectory::TimedStatePath MotionPlanning::getReplay(double dt)
{
    return createReplay(this->workCell_, this->path_, this->robot_, dt);
}

void MotionPlanning::getReplay(rw::trajectory::TimedStatePath& replay, double dt)
{
    replay = createReplay(this->workCell_, this->path_, this->robot_, dt);
}


